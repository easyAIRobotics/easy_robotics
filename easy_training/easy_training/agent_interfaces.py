import os
import threading
import time
from enum import Enum
from rclpy.callback_groups import ReentrantCallbackGroup

import rclpy
from rclpy.node import Node

from easy_training.utils import LossVisualizer


class AgentMode(Enum):
    IDLE = 0
    SELF_LEARNING = 1
    DETERMINISTIC_POLICY = 2
    BEHAVIOR_CLONING = 3
    IDLE_UPDATING = 4
    PERFORMING = 5
    
    
class ReplayBuffer:
    def __init__(self, capacity, device="cuda"):
        self.capacity = capacity
        self.device = device
        self.ptr = 0
        self.buffer_size = 0
    
    def save_to_disk(self, file_path: str):
        raise NotImplementedError("[ReplayBuffer] The save_to_disk method must be implemented by the subclass.")
        
    def load_from_disk(self, file_path: str):
        raise NotImplementedError("[ReplayBuffer] The load_from_disk method must be implemented by the subclass.")
    
    
class StateInterface:
    def __init__(self, node: Node):
        self._node = node
        self._mode = AgentMode.IDLE
        self.action = None
        self.state_interface_callback_group = ReentrantCallbackGroup()

    def check_sanity(self) -> bool:
        raise NotImplementedError("[StateInterface] The check_sanity method must be implemented by the subclass.")
    
    def set_action(self, action: str):
        raise NotImplementedError("[StateInterface] The set_action method must be implemented by the subclass.")
        
    def set_mode(self, mode: AgentMode):
        self._mode = mode
        self._node.get_logger().info(f"[SkillExecutionStateInterface] Mode set to: {self._mode.name}")
        # if self._mode == AgentMode.IDLE:
        #     self.selected_bbox = None
    
        # Get current states and observations
    def get_state(self) -> dict:
        raise NotImplementedError("[StateInterface] The get_state method must be implemented by the subclass.")
    
    def get_reward(self) -> float:
        raise NotImplementedError("[StateInterface] The get_reward method must be implemented by the subclass.")


class ActionInterface:
    def __init__(self, node: Node):
        self._node = node
        self.frequency = 3.0  # Default frequency for action execution
        self.action_interface_callback_group = ReentrantCallbackGroup()
        self._mode = AgentMode.IDLE
        self.action = None
        
    def set_action(self, action: str):
        self.action = action
        self._node.get_logger().info(f"[ActionInterface] Action set to: {self.action}")
        
    def set_mode(self, mode: AgentMode):
        self._mode = mode
        self._node.get_logger().info(f"[SkillExecutionStateInterface] Mode set to: {self._mode.name}")
        if self._mode == AgentMode.IDLE:
            self.selected_bbox = None
    
    # Perform action and return the reward"""
    def perform(self, act_vec: list, state_interface: StateInterface) -> tuple[float, dict]:
        raise NotImplementedError("[ActionInterface] The perform method must be implemented by the subclass.")
    
    def wait_for_next_state(self):
        # Wait for the next state update based on the specified frequency
        time.sleep(1.0 / self.frequency)
    

class AgentInterface:
    def __init__(self, node: Node,
                 action_interface: ActionInterface, 
                 state_interface: StateInterface):
        self._node = node
        
        self.action_interface = action_interface
        self.state_interface = state_interface
        self.sac_agent = None
        
        self.mode_ = AgentMode.IDLE
        
        self.rl_replay_buffer = None
        self.bc_replay_buffer = None
        self.past_bc_replay_buffer = None
        self.past_rl_replay_buffer = None
        self.validate_buffer = None
        self.loss_visualizer = LossVisualizer()
        
        self._node.declare_parameter("storage_path", "/tmp/easy_training_data")
        self.storage_path = self._node.get_parameter("storage_path").get_parameter_value().string_value
        
        self.buffer_folder = self.storage_path + "/buffer_latest"
        self.validate_folder = self.storage_path + "/validate_latest"
        self.model_folder = self.storage_path + "/model_latest"
            
        self._update_lock = threading.Lock()
        self._update_thread = None
        self._buffer_lock = threading.Lock()
        
    def train_loop(self):
        while rclpy.ok():
            reward = 0.0
                        
            if self.mode_ == AgentMode.IDLE:
                time.sleep(1.0)
                continue
            
            if self.mode_ == AgentMode.IDLE_UPDATING:
                self.update()
                time.sleep(0.05)
                continue
            
            if not self.state_interface.check_sanity():
                self.action_interface.wait_for_next_state()
                continue
            
            if self.mode_ == AgentMode.DETERMINISTIC_POLICY:
                action = self.infer_action(deterministic=False)
                if not action:
                    continue
                _, _ = self.action_interface.perform(action, self.state_interface)
                # self.update()
                continue
            
            if self.mode_ == AgentMode.SELF_LEARNING:
                action = self.infer_action(deterministic=False)
                if not action:
                    continue
                reward, transition = self.action_interface.perform(action, self.state_interface)
                if transition:
                    transition["reward"] = reward  # Use the reward from performing the policy action
                    self.add_rl_transition(transition)
                self.update()
                
            if self.mode_ == AgentMode.BEHAVIOR_CLONING:
                reward, transition = self.action_interface.perform([], self.state_interface)
                if transition:
                    transition["reward"] = reward  # Use the reward from performing the user action
                    self.add_bc_transition(transition)
                self.update()
                
            if self.mode_ == AgentMode.PERFORMING:
                action = self.infer_action(deterministic=True)
                if not action:
                    continue
                _, _ = self.action_interface.perform(action, self.state_interface)
            
            print(f"[AgentInterface] Received reward: {reward}", flush=True)
        self.reset()
        
        
    def set_mode(self, mode: str):
        print(f"[AgentInterface] Setting mode to: {mode}", flush=True)
        if mode == "training/auto":
            self.mode_ = AgentMode.SELF_LEARNING
        elif mode == "training/exec":
            self.mode_ = AgentMode.DETERMINISTIC_POLICY
        elif mode == "training/manual":
            self.mode_ = AgentMode.BEHAVIOR_CLONING
        elif mode == "training/stop":
            self.mode_ = AgentMode.IDLE
        elif mode == "training/idle":
            self.mode_ = AgentMode.IDLE_UPDATING
        elif mode == "performing":
            self.mode_ = AgentMode.PERFORMING
        else:
            print(f"[AgentInterface] Unknown mode: {mode}. Defaulting to IDLE.", flush=True)
            self.mode_ = AgentMode.IDLE
            
        self.action_interface.set_mode(self.mode_)
        self.state_interface.set_mode(self.mode_)
        
    
    def execute_command(self, command: str):
        print(f"[AgentInterface] Executing command: {command}", flush=True)
        if command == "training/save_buffer":
            stamped_buffer_folder = self.storage_path + f"/buffer_latest/data_{int(time.time())}"
            if not os.path.exists(stamped_buffer_folder):
                os.makedirs(stamped_buffer_folder)
            self.rl_replay_buffer.save_to_disk(stamped_buffer_folder + "/rl_replay_buffer.npz")
            self.bc_replay_buffer.save_to_disk(stamped_buffer_folder + "/bc_replay_buffer.npz")
            
        elif command == "training/save_model":
            if self.sac_agent is not None:
                stamped_model_folder = self.storage_path + f"/model_{int(time.time())}"
                if not os.path.exists(stamped_model_folder):
                    os.makedirs(stamped_model_folder)
                self.sac_agent.save_model(stamped_model_folder)
        else:
            print(f"[AgentInterface] Unknown command: {command}", flush=True)
        print(f"[AgentInterface] Command execution completed: {command}", flush=True)
    
    
    def infer_action(self, deterministic=True) -> dict:
        raise NotImplementedError("[AgentInterface] The infer_action method must be implemented by the subclass.")
    
    
    def update(self):
        raise NotImplementedError("[AgentInterface] The update method must be implemented by the subclass.")
    
        
    def reset(self):
        raise NotImplementedError("[AgentInterface] The reset method must be implemented by the subclass.")
    
    
    def add_rl_transition(self, transition: dict):
        raise NotImplementedError("[AgentInterface] The add_rl_transition method must be implemented by the subclass.")
            

    def add_bc_transition(self, transition: dict):
        raise NotImplementedError("[AgentInterface] The add_bc_transition method must be implemented by the subclass.")
    
    def __del__(self):
        self.loss_visualizer.close()
