from easy_training.agent_interfaces import AgentInterface, ActionInterface, StateInterface
from easy_training.skill_execution.skill_execution_action_interface import SkillExecutionActionInterface
from easy_training.skill_execution.skill_execution_state_interface import SkillExecutionStateInterface

import rclpy
from rclpy.node import Node
from easy_interfaces.srv import SetString

import random


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
        
        
    def set_action_callback(self, request, response):
        self._node.get_logger().info(f"Received request to set action to: {request.data}")
        self.action = request.data
        self.action_interface.set_action(self.action)
        response.success = True
        response.message = f"Agent action set to {request.data}"
        return response
        
    def infer_action(self, state):
        eef_pose = state["eef_pose"]
        random_vector = [random.gauss(0.0, 0.001), random.gauss(0.0, 0.001), random.gauss(0.0, 0.001), 0.0, 0.0, 0.0, 0.0]
        
        suction_command = state["cmd_suction_state"]
        
        if random.random() < 0.1:
            suction_command = 1.0 - suction_command
        
        act = {
            "eef_pose": [eef_pose_val + rand_val for eef_pose_val, rand_val in zip(eef_pose, random_vector)],
            "suction_command": suction_command
        }
                
        return act
    
    def update(self):
        print(f"[SkillExecutionAgentInterface] Updating agent based on replay buffer with {len(self.replay_buffer)} samples", flush=True)
        
        
    def reset(self):
        print(f"[SkillExecutionAgentInterface] Resetting agent state", flush=True)
