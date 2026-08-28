import time
import os

from easy_training.task_planning.task_planning_memory import TaskPlanningReplayBuffer
from easy_training.task_planning.task_planning_transformers import VisionTransformer

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class HandImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 4, kernel_size=3, stride=2, padding=1)  # 64x64 -> 32x32
        self.conv2 = nn.Conv2d(4, 8, kernel_size=3, stride=2, padding=1)    # 32x32 -> 16x16
        self.conv3 = nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1)   # 16x16 -> 8x8
        self.conv4 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)  # 8x8 -> 4x4
        self.fc = nn.Linear(32 * 4 * 4, 64)  # Assuming input image size is 64x64

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        return x

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = ConvBlock(
            in_channels,
            out_channels
        )

    def forward(self, x):
        x = F.interpolate(
            x,
            scale_factor=2,
            mode="bilinear",
            align_corners=False
        )

        x = self.conv(x)

        return x


class AttentionMapDecoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.up1 = UpBlock(384, 192)  # 14 -> 28
        self.up2 = UpBlock(192, 96)   # 28 -> 56
        self.up3 = UpBlock(96, 48)    # 56 -> 112
        self.up4 = UpBlock(48, 24)    # 112 -> 224

        self.head = nn.Conv2d(
            24,
            1,
            kernel_size=1
        )

    def forward(self, x):
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)

        heatmap = self.head(x)

        return heatmap
    
class TaskPlanningPolicyNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        # Skill decoder
        self.skill_decoder = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 3)  # Output skill vector of size 3
        )
        
        # Hand image encoder
        self.hand_image_encoder = HandImageEncoder()

        # Project robot state to token dimension
        self.state_proj = nn.Linear(1, 384)
        self.state_norm = nn.LayerNorm(384)
        
        self.cls_norm = nn.LayerNorm(384)
        self.patch_norm = nn.LayerNorm(384)
        
        self.patch_fusion = nn.Sequential(
            nn.Linear(384 * 2 + 64, 384),
            nn.ReLU()
        )

        # heatmap decoder
        self.heatmap_decoder = AttentionMapDecoder()

    def forward(
        self,
        cls_tokens,
        patch_tokens,
        hand_image,
        robot_state
    ):
        """
        cls_tokens:   (B, 384)
        patch_tokens: (B, 196, 384)
        hand_image:    (B, 3, 64, 64)
        robot_state:  (B, 1)

        Returns:
            skill_vector: (B, 3)
            heatmap:      (B, 1, 224, 224)
        """
        
        hand_features = self.hand_image_encoder(hand_image)  # (B, 64)

        # ----------------------------
        # Skill branch
        # ----------------------------
        cls_feature = self.cls_norm(cls_tokens)
        state_feature = self.state_proj(robot_state)
        state_feature = self.state_norm(state_feature)
        
        skill_input = torch.cat(
            (cls_feature, state_feature),
            dim=-1
        )  # (B, 384 + 384 = 768)

        skill_vector = self.skill_decoder(
            skill_input
        )

        # ----------------------------
        # heatmap branch
        # ----------------------------
        patch_tokens = self.patch_norm(patch_tokens)
        patch_input = torch.cat([
            patch_tokens,
            state_feature.unsqueeze(1).expand(-1, 196, -1),
            hand_features.unsqueeze(1).expand(-1, 196, -1)
        ], dim=-1)
        patch_tokens = self.patch_fusion(patch_input)  # (B, 196, 384)

        B = patch_tokens.shape[0]

        x = patch_tokens.reshape(
            B,
            14,
            14,
            384
        )

        x = x.permute(
            0,
            3,
            1,
            2
        )  # (B,384,14,14)

        heatmap = self.heatmap_decoder(x)

        return skill_vector, heatmap

    
    
class TaskPlanningSACAgent:
    def __init__(self, node):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = TaskPlanningPolicyNetwork().to(self.device)
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=3e-4)
        self.vision_transformer = VisionTransformer()
        self.replay_buffer = TaskPlanningReplayBuffer()


    def infer_action(self, state: dict, deterministic=True):
        print(f"------------- Robot state: {state['robot_state']}", flush=True)
        self.policy.eval()
        with torch.no_grad():
            rgb_image = torch.from_numpy(state["rgb_image"]).unsqueeze(0).to(self.device)
            hand_image = torch.from_numpy(state["rgb_image_hand"]).unsqueeze(0).to(self.device)
            hand_image = hand_image / 255.0
            hand_image = hand_image.permute(0, 3, 1, 2)  # (B, 3, 64, 64)
            robot_state = torch.tensor(state["robot_state"]).unsqueeze(0).to(self.device)
            cls_token, patch_tokens = self.vision_transformer.extract_features(rgb_image)
            skill_vector, heatmap = self.policy.forward(cls_token, patch_tokens, hand_image, robot_state)
            # normalize heatmap to [0,1] using sigmoid
            heatmap = torch.sigmoid(heatmap)
        print(f"------------- Predicted skill vector: {skill_vector.cpu().numpy()}", flush=True)
        return skill_vector.cpu().numpy(), heatmap.cpu().numpy()
    
    
    def update(self, bc_samples, past_bc_samples, val_samples):
        # Merge all samples into one batch for policy update
        if bc_samples is None and past_bc_samples is None:
            return
        
        if bc_samples is not None:
            rgb_image, hand_image, heatmap, robot_state, skill, reward, done = bc_samples
            if past_bc_samples is not None:
                past_rgb_image, past_hand_image, past_heatmap, past_robot_state, past_skill, past_reward, past_done = past_bc_samples
                rgb_image = torch.cat((rgb_image, past_rgb_image), dim=0)
                hand_image = torch.cat((hand_image, past_hand_image), dim=0)
                heatmap = torch.cat((heatmap, past_heatmap), dim=0)
                robot_state = torch.cat((robot_state, past_robot_state), dim=0)
                skill = torch.cat((skill, past_skill), dim=0)
                reward = torch.cat((reward, past_reward), dim=0)
                done = torch.cat((done, past_done), dim=0)

        elif past_bc_samples is not None:
            rgb_image, hand_image, heatmap, robot_state, skill, reward, done = past_bc_samples
            if bc_samples is not None:
                bc_rgb_image, bc_hand_image, bc_heatmap, bc_robot_state, bc_skill, bc_reward, bc_done = bc_samples
                rgb_image = torch.cat((rgb_image, bc_rgb_image), dim=0)
                hand_image = torch.cat((hand_image, bc_hand_image), dim=0)
                heatmap = torch.cat((heatmap, bc_heatmap), dim=0)
                robot_state = torch.cat((robot_state, bc_robot_state), dim=0)
                skill = torch.cat((skill, bc_skill), dim=0)
                reward = torch.cat((reward, bc_reward), dim=0)
                done = torch.cat((done, bc_done), dim=0)
        
        self.policy.train()
        print(f"Image shape: {rgb_image.shape}", flush=True)
        cls_token, patch_tokens = self.vision_transformer.extract_features(rgb_image)
        # Normalize hand_image to [0,1]
        hand_image = hand_image / 255.0
        hand_image = hand_image.permute(0, 3, 1, 2)  # (B, 3, 64, 64)
        
        # Policy update using behavior cloning loss
        pred_skill_vector, pred_heatmap = self.policy.forward(cls_token, patch_tokens, hand_image, robot_state)
        skill_loss = F.mse_loss(pred_skill_vector, skill)
        ref_heatmap = heatmap.unsqueeze(1)  # Add channel dimension
        heatmap_loss = F.binary_cross_entropy_with_logits(pred_heatmap, ref_heatmap)
        
        total_loss = skill_loss + 10 * heatmap_loss
        self.policy_optimizer.zero_grad()
        total_loss.backward()
        self.policy_optimizer.step()
                
        print(f"Policy update - Total Loss: {total_loss.item():.4f}, Skill Loss: {skill_loss.item():.4f}, heatmap Loss: {heatmap_loss.item():.4f}", flush=True)
        
        # Compute validate loss if validation samples are provided
        if val_samples is not None:
            val_rgb_image, val_hand_image, val_heatmap, val_robot_state, val_skill, val_reward, val_done = val_samples
            self.policy.eval()
            with torch.no_grad():
                val_cls_token, val_patch_tokens = self.vision_transformer.extract_features(val_rgb_image)
                val_hand_image = val_hand_image / 255.0
                val_hand_image = val_hand_image.permute(0, 3, 1, 2)  # (B, 3, 64, 64)
                val_pred_skill_vector, val_pred_heatmap = self.policy.forward(val_cls_token, val_patch_tokens, val_hand_image, val_robot_state)
                val_skill_loss = F.mse_loss(val_pred_skill_vector, val_skill)
                val_ref_heatmap = val_heatmap.unsqueeze(1)  # Add channel dimension
                val_heatmap_loss = F.binary_cross_entropy_with_logits(val_pred_heatmap, val_ref_heatmap)
                val_total_loss = val_skill_loss + 10 * val_heatmap_loss
            print(f"Validation - Total Loss: {val_total_loss.item():.4f}, Skill Loss: {val_skill_loss.item():.4f}, heatmap Loss: {val_heatmap_loss.item():.4f}", flush=True)
        else:
            val_total_loss = None
            val_skill_loss = None
            val_heatmap_loss = None
            
        return {
            "total_loss": total_loss.item(),
            "skill_loss": skill_loss.item(),
            "heatmap_loss": heatmap_loss.item(),
            "val_total_loss": val_total_loss.item() if val_total_loss is not None else 0.0,
            "val_skill_loss": val_skill_loss.item() if val_skill_loss is not None else 0.0,
            "val_heatmap_loss": val_heatmap_loss.item() if val_heatmap_loss is not None else 0.0
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
