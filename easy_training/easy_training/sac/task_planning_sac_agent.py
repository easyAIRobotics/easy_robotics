import time
import os

from easy_training.task_planning.task_planning_memory import TaskPlanningReplayBuffer
from easy_training.task_planning.task_planning_transformers import VisionTransformer

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
    
class TaskPlanningPolicyNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        # Skill decoder
        self.skill_decoder = nn.Sequential(
            nn.Linear(384 + 1, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )

        # Project robot state to token dimension
        self.state_proj = nn.Linear(1, 384)
        self.skill_proj = nn.Linear(3, 384)

        # Heatmap decoder
        self.heatmap_decoder = nn.Sequential(
            nn.Conv2d(384, 128, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv2d(64, 1, kernel_size=1)
        )

    def forward(
        self,
        cls_tokens,
        patch_tokens,
        robot_state
    ):
        """
        cls_tokens:   (B, 384)
        patch_tokens: (B, 196, 384)
        robot_state:  (B, 1)

        Returns:
            skill_vector: (B, 3)
            heatmap:      (B, 1, 224, 224)
        """

        # ----------------------------
        # Skill branch
        # ----------------------------

        skill_input = torch.cat(
            [cls_tokens, robot_state],
            dim=1
        )

        skill_vector = self.skill_decoder(
            skill_input
        )

        # ----------------------------
        # Heatmap branch
        # ----------------------------

        # Inject robot state into every patch token
        state_feature = self.state_proj(
            robot_state
        )                           # (B, 384)
        
        skill_feature = self.skill_proj(
            skill_vector
        )                           # (B, 384)

        patch_tokens = (
            patch_tokens
            + state_feature.unsqueeze(1)
            + skill_feature.unsqueeze(1)
        )                           # (B,196,384)

        B = patch_tokens.shape[0]

        # 196 = 14 x 14
        x = patch_tokens.reshape(
            B,
            14,
            14,
            384
        )

        x = x.permute(
            0, 3, 1, 2
        )                           # (B,384,14,14)

        heatmap = self.heatmap_decoder(
            x
        )                           # (B,1,14,14)

        heatmap = F.interpolate(
            heatmap,
            size=(224, 224),
            mode="bilinear",
            align_corners=False
        )

        return skill_vector, heatmap

    
    
class TaskPlanningSACAgent:
    def __init__(self, node):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = TaskPlanningPolicyNetwork().to(self.device)
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=3e-4)
        self.vision_transformer = VisionTransformer()
        self.replay_buffer = TaskPlanningReplayBuffer()


    def infer_action(self, state: dict, deterministic=True):
        self.policy.eval()
        with torch.no_grad():
            rgb_image = torch.from_numpy(state["rgb_image"]).unsqueeze(0).to(self.device)
            robot_state = torch.tensor(state["robot_state"]).unsqueeze(0).to(self.device)
            cls_token, patch_tokens = self.vision_transformer.extract_features(rgb_image)
            skill_vector, heatmap = self.policy.forward(cls_token, patch_tokens, robot_state)

        return skill_vector.cpu().numpy(), heatmap.cpu().numpy()
    
    
    def update(self, bc_samples, past_bc_samples, val_samples):
        # Merge all samples into one batch for policy update
        if bc_samples is None and past_bc_samples is None:
            return
        
        if bc_samples is not None:
            rgb_image, heatmap, robot_state, skill, reward, done = bc_samples
            if past_bc_samples is not None:
                past_rgb_image, past_heatmap, past_robot_state, past_skill, past_reward, past_done = past_bc_samples
                rgb_image = torch.cat((rgb_image, past_rgb_image), dim=0)
                heatmap = torch.cat((heatmap, past_heatmap), dim=0)
                robot_state = torch.cat((robot_state, past_robot_state), dim=0)
                skill = torch.cat((skill, past_skill), dim=0)
                reward = torch.cat((reward, past_reward), dim=0)
                done = torch.cat((done, past_done), dim=0)

        elif past_bc_samples is not None:
            rgb_image, heatmap, robot_state, skill, reward, done = past_bc_samples
            if bc_samples is not None:
                bc_rgb_image, bc_heatmap, bc_robot_state, bc_skill, bc_reward, bc_done = bc_samples
                rgb_image = torch.cat((rgb_image, bc_rgb_image), dim=0)
                heatmap = torch.cat((heatmap, bc_heatmap), dim=0)
                robot_state = torch.cat((robot_state, bc_robot_state), dim=0)
                skill = torch.cat((skill, bc_skill), dim=0)
                reward = torch.cat((reward, bc_reward), dim=0)
                done = torch.cat((done, bc_done), dim=0)
        
        cls_token, patch_tokens = self.vision_transformer.extract_features(rgb_image)    
        
        # Policy update using behavior cloning loss
        pred_skill_vector, pred_heatmap = self.policy.forward(cls_token, patch_tokens, robot_state)
        skill_loss = F.mse_loss(pred_skill_vector, skill)
        heatmap_loss = F.mse_loss(pred_heatmap, heatmap.unsqueeze(1))
        total_loss = 100 * skill_loss + heatmap_loss
        self.policy_optimizer.zero_grad()
        total_loss.backward()
        self.policy_optimizer.step()
        
        print(f"Policy update - Total Loss: {total_loss.item():.4f}, Skill Loss: {skill_loss.item():.4f}, Heatmap Loss: {heatmap_loss.item():.4f}", flush=True)
        
        return {
            "total_loss": total_loss.item(),
            "skill_loss": skill_loss.item(),
            "heatmap_loss": heatmap_loss.item()
        }
        
    
    def load_model(self, model_folder):
        model_path = os.path.join(model_folder, "sac_agent.pth")
        checkpoint = torch.load(model_path, map_location=self.device)
        try:
            self.policy.load_state_dict(checkpoint["policy"])
        except Exception:
            print("[SACAgent] Warning: Failed to load policy state dict")
        try:
            self.policy_optimizer.load_state_dict(checkpoint["policy_optimizer"])
        except Exception:
            print("[SACAgent] Warning: Failed to load policy optimizer state dict")

    
    def save_model(self, model_folder):
        model_path = os.path.join(model_folder, "sac_agent.pth")
        torch.save({
            "policy": self.policy.state_dict(),
            "policy_optimizer": self.policy_optimizer.state_dict(),
        }, model_path)
