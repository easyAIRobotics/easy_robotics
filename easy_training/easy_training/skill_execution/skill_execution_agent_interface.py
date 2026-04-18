import os
import threading
import time

from easy_training.agent_interfaces import AgentInterface, ActionInterface, StateInterface
from easy_training.skill_execution.skill_execution_action_interface import SkillExecutionActionInterface
from easy_training.skill_execution.skill_execution_state_interface import SkillExecutionStateInterface
from easy_training.skill_execution.skill_execution_memory import SkillExecutionReplayBuffer
from easy_training.sac.skill_execution_sac_agent import SkillExecutionSACAgent

import rclpy
from rclpy.node import Node
from easy_interfaces.srv import SetString

import random

RL_BUFFER_CAPACITY = 10000
BC_BUFFER_CAPACITY = 50000
VAL_BUFFER_CAPACITY = 2000


class SkillExecutionAgentInterface(AgentInterface):
    def __init__(self, node: Node, agent_name: str = "skill_execution"):
        super().__init__(
            node, 
            SkillExecutionActionInterface(node), 
            SkillExecutionStateInterface(node)
        )
        
        self.set_action_service = self._node.create_service(
            srv_type=SetString,
            srv_name=f"{agent_name}/set_action",
            callback=self.set_action_callback
        )
        self.action = "pick"
        self.action_interface.set_action(self.action)
        
        self.rl_replay_buffer = SkillExecutionReplayBuffer(
            capacity=RL_BUFFER_CAPACITY,
            image_shape=(30, 40, 3),
            skill_dim=3,
            robot_state_dim=16,
            action_dim=7,
            device="cuda"
        )
        
        self.bc_replay_buffer = SkillExecutionReplayBuffer(
            capacity=BC_BUFFER_CAPACITY,
            image_shape=(30, 40, 3),
            skill_dim=3,
            robot_state_dim=16,
            action_dim=7,
            device="cuda"
        )
        
        self.validate_buffer = SkillExecutionReplayBuffer(
            capacity=VAL_BUFFER_CAPACITY,
            image_shape=(30, 40, 3),
            skill_dim=3,
            robot_state_dim=16,
            action_dim=7,
            device="cuda"
        )
        
        self.sac_agent = SkillExecutionSACAgent(node)
        
        if os.path.exists(self.buffer_folder):
            self._node.get_logger().info(f"[SkillExecutionAgentInterface] Loading replay buffers from {self.buffer_folder}...")
            self.rl_replay_buffer.load_from_disk(self.buffer_folder + "/rl_replay_buffer.npz")
            self.bc_replay_buffer.load_from_disk(self.buffer_folder + "/bc_replay_buffer.npz")
            self._node.get_logger().info(f"[SkillExecutionAgentInterface] Loaded {self.rl_replay_buffer.size()} RL samples and {self.bc_replay_buffer.size()} BC samples from disk.")
        else:
            self._node.get_logger().info(f"[SkillExecutionAgentInterface] No existing replay buffer found at {self.buffer_folder}, starting with empty buffers.")
        
        if os.path.exists(self.validate_folder):
            self._node.get_logger().info(f"[SkillExecutionAgentInterface] Loading validation buffer from {self.validate_folder}...")
            self.validate_buffer.load_from_disk(self.validate_folder + "/bc_replay_buffer.npz")
            self._node.get_logger().info(f"[SkillExecutionAgentInterface] Loaded {self.validate_buffer.size()} validation samples from disk.")
        else:
            self._node.get_logger().info(f"[SkillExecutionAgentInterface] No existing validation buffer found at {self.validate_folder}, starting with empty validation buffer.")
            
        if os.path.exists(self.model_folder):
            self._node.get_logger().info(f"[SkillExecutionAgentInterface] Loading SAC agent model from {self.model_folder}...")
            self.sac_agent.load_model(self.model_folder)
        else:
            self._node.get_logger().info(f"[SkillExecutionAgentInterface] No existing model found at {self.model_folder}, starting with new agent.")
        
        
    def set_action_callback(self, request, response):
        self._node.get_logger().info(f"Received request to set action to: {request.data}")
        self.action = request.data
        self.action_interface.set_action(self.action)
        self.state_interface.set_action(self.action)
        response.success = True
        response.message = f"Agent action set to {request.data}"
        return response
    

    def infer_action(self, deterministic=True):
        state_dict = {
            "image": self.state_interface.get_image(),
            "target_image": self.state_interface.get_target_image(),
            "original_target_image": self.state_interface.get_original_target_image(),
            "skill": self.state_interface.get_skill(),
            "robot_state": self.state_interface.get_robot_state()
        }
        action = self.sac_agent.infer_action(state_dict, deterministic)
        
        if action is not None:
            action = action.tolist()[0]
        else:
            action = []

        return action

    def update(self):
        # Return if an update is already in progress
        if self._update_thread is not None and self._update_thread.is_alive():
            return
        
        with self._update_lock:
            self._update_thread = threading.Thread(
                target=self._update_worker,
                daemon=True
            )
            self._update_thread.start()

    def _update_worker(self):
        try:
            with self._buffer_lock:
                rl_batch = self.rl_replay_buffer.sample(128, recent=True)
                bc_batch = self.bc_replay_buffer.sample(1024, recent=False)
                val_batch = self.validate_buffer.sample(128)

            losses = self.sac_agent.update(rl_batch, bc_batch, val_batch)
            self.loss_visualizer.update(losses)

        except Exception as e:
            self._node.get_logger().error(f"[SkillExecutionAgentInterface] Update failed: {e}")
        
        
    def reset(self):
        self._node.get_logger().info(f"[SkillExecutionAgentInterface] Resetting agent state")
        
    def add_rl_transition(self, transition: dict):
        with self._buffer_lock:
            # self._node.get_logger().info(
            #     f"[SkillExecutionAgentInterface] Adding RL transition to replay buffer, action taken: {transition['action']}, reward: {transition['reward']}"
            # )
            self.rl_replay_buffer.add(
                image=transition["image"],
                target_image=transition["target_image"],
                original_target_image=transition["original_target_image"],
                skill=transition["skill"],
                robot_state=transition["robot_state"],
                j_action=transition["j_action"],
                e_action=transition["e_action"],
                reward=transition["reward"],
                done=transition["done"],
                next_image=transition["next_image"],
                next_target_image=transition["next_target_image"],
                next_original_target_image=transition["next_original_target_image"],
                next_skill=transition["next_skill"],
                next_robot_state=transition["next_robot_state"],
            )
            self._node.get_logger().info(f"[SkillExecutionAgentInterface] RL Buffer size: {self.rl_replay_buffer.size()}")
        
    def add_bc_transition(self, transition: dict):
        with self._buffer_lock:
            # self._node.get_logger().info(
            #     f"[SkillExecutionAgentInterface] Adding BC transition to replay buffer, j_action taken: {transition['j_action']}, e_action taken: {transition['e_action']}, reward: {transition['reward']}"
            # )
            self.bc_replay_buffer.add(
                image=transition["image"],
                target_image=transition["target_image"],
                original_target_image=transition["original_target_image"],
                skill=transition["skill"],
                robot_state=transition["robot_state"],
                j_action=transition["j_action"],
                e_action=transition["e_action"],
                reward=transition["reward"],
                done=transition["done"],
                next_image=transition["next_image"],
                next_target_image=transition["next_target_image"],
                next_original_target_image=transition["next_original_target_image"],
                next_skill=transition["next_skill"],
                next_robot_state=transition["next_robot_state"],
            )
            self._node.get_logger().info(f"[SkillExecutionAgentInterface] BC Buffer size: {self.bc_replay_buffer.size()}")
