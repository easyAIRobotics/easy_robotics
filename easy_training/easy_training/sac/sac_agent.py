import os

from rclpy.node import Node
import torch

class SACAgent:
    def __init__(self, node: Node, agent_name: str):
        self.node = node
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.node.declare_parameter(f"{agent_name}.model_path", "")
        self.model_path = self.node.get_parameter(f"{agent_name}.model_path").get_parameter_value().string_value
        
        self.encoder = None
        self.policy = None
        self.q1 = None
        self.q2 = None
        self.target_q1 = None
        self.target_q2 = None
        
        self.policy_optimizer = None
        self.q1_optimizer = None
        self.q2_optimizer = None
        self.encoder_optimizer = None
    
    
    def infer_action(self, state, deterministic=True):
        raise NotImplementedError("SACAgent infer_action method not implemented")
    
    def update(self, rl_samples, bc_samples):
        raise NotImplementedError("SACAgent update method not implemented")
    
    
    def load(self):
        checkpoint = torch.load(self.model_path, map_location=self.device)

        self.encoder.load_state_dict(checkpoint["encoder"])
        self.policy.load_state_dict(checkpoint["policy"])
        self.q1.load_state_dict(checkpoint["q1"])
        self.q2.load_state_dict(checkpoint["q2"])
        self.target_q1.load_state_dict(checkpoint["target_q1"])
        self.target_q2.load_state_dict(checkpoint["target_q2"])
        
    
    def save(self):
        torch.save({
            "encoder": self.encoder.state_dict(),
            "policy": self.policy.state_dict(),
            "q1": self.q1.state_dict(),
            "q2": self.q2.state_dict(),
            "target_q1": self.target_q1.state_dict(),
            "target_q2": self.target_q2.state_dict(),
        }, self.model_path)
