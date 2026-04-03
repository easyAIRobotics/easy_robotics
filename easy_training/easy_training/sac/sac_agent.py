import os

from rclpy.node import Node
import torch

class SACAgent:
    def __init__(self, node: Node, agent_name: str):
        self._node = node
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.policy = None
        self.q = None
        self.target_q = None
        
        self.policy_optimizer = None
        self.q_optimizer = None
    
    
    def infer_action(self, state, deterministic=True):
        raise NotImplementedError("SACAgent infer_action method not implemented")
    
    def update(self, rl_samples, bc_samples, val_samples):
        raise NotImplementedError("SACAgent update method not implemented")
    
    
    def load_model(self, model_folder):
        model_path = os.path.join(model_folder, "sac_agent.pth")
        checkpoint = torch.load(model_path, map_location=self.device)
        try:
            self.policy.load_state_dict(checkpoint["policy"])
        except Exception:
            print("[SACAgent] Warning: Failed to load policy state dict")
        try:
            self.q.load_state_dict(checkpoint["q"])
        except Exception:
            print("[SACAgent] Warning: Failed to load q state dict")
        try:
            self.target_q.load_state_dict(checkpoint["target_q"])
        except Exception:
            print("[SACAgent] Warning: Failed to load target_q state dict")
        try:
            self.policy_optimizer.load_state_dict(checkpoint["policy_optimizer"])
        except Exception:
            print("[SACAgent] Warning: Failed to load policy optimizer state dict")
        try:
            self.q_optimizer.load_state_dict(checkpoint["q_optimizer"])
        except Exception:
            print("[SACAgent] Warning: Failed to load q optimizer state dict")

    
    def save_model(self, model_folder):
        model_path = os.path.join(model_folder, "sac_agent.pth")
        torch.save({
            "policy": self.policy.state_dict(),
            "q": self.q.state_dict(),
            "target_q": self.target_q.state_dict(),
            "policy_optimizer": self.policy_optimizer.state_dict(),
            "q_optimizer": self.q_optimizer.state_dict(),
        }, model_path)
