import time
import os

from easy_training.task_planning.task_planning_memory import TaskPlanningReplayBuffer
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from easy_training.task_planning.task_planning_cfg import NUM_HEADS, TEXT_EMBEDDING_DIM, SKILL_VOCAB


class ImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, kernel_size=5, stride=2) # 224x224 -> 110x110
        self.conv2 = nn.Conv2d(8, 16, kernel_size=5, stride=2) # 110x110 -> 53x53
        self.conv3 = nn.Conv2d(16, 32, kernel_size=5, stride=2) # 53x53 -> 25x25
        self.conv4 = nn.Conv2d(32, 64, kernel_size=5, stride=2) # 25x25 -> 11x11
        self.conv5 = nn.Conv2d(64, 128, kernel_size=5, stride=2) # 11x11 -> 4x4
        self.fc = nn.Linear(128 * 4 * 4, 256)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc(x))
        return x

class BoundingboxHeadEncoder(nn.Module):
    def __init__(self, bbox_dim = 4, text_embedding_dim = TEXT_EMBEDDING_DIM):
        super().__init__()
        self.bbox_fc = nn.Linear(bbox_dim, 16)
        self.text_fc = nn.Linear(text_embedding_dim, 256)
        self.fc1 = nn.Linear(16 + 256, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 256)
        self.fc5 = nn.Linear(256, 64)
        
    def forward(self, bbox, text):
        bbox = F.relu(self.bbox_fc(bbox))
        text = F.relu(self.text_fc(text))
        x = torch.cat((bbox, text), dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x)) + x
        x = F.relu(self.fc3(x)) + x
        x = F.relu(self.fc4(x)) + x
        x = F.relu(self.fc5(x))
        return x
    
    
class TaskPlanningPolicyNetwork(nn.Module):
    def __init__(self, num_heads=NUM_HEADS, text_embedding_dim=TEXT_EMBEDDING_DIM, robot_state_dim=1, action_dim=NUM_HEADS + 3):
        super().__init__()
        self.image_encoder = ImageEncoder()
        self.bbox_encoder = BoundingboxHeadEncoder()
        self.fc1 = nn.Linear(256 + num_heads * 64 + robot_state_dim, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 1024)
        self.fc4 = nn.Linear(1024, 512)
        self.fc5 = nn.Linear(512, action_dim)
        
    def forward(self, rgb_image, bbox_list, class_list, robot_state):
        batch_size = bbox_list.size(0)
        rgb_image = rgb_image.float() / 255.0
        img_features = self.image_encoder(rgb_image.permute(0, 3, 1, 2)) # (B, 3, H, W) -> (B, 256)
        encoded_bboxes = []
        for i in range(NUM_HEADS):
            encoded_bbox = self.bbox_encoder(bbox_list[:, i], class_list[:, i])
            encoded_bboxes.append(encoded_bbox)
        encoded_bboxes = torch.cat(encoded_bboxes, dim=-1)
        
        x = torch.cat((img_features, encoded_bboxes, robot_state), dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x)) + x
        x = F.relu(self.fc3(x)) + x
        x = F.relu(self.fc4(x))
        
        # Softmax for bbox selection, sigmoid for skill selection
        action = self.fc5(x)
        
        action[:, :NUM_HEADS] = F.softmax(action[:, :NUM_HEADS], dim=-1)
        
        return action
    
    
class TaskPlanningSACAgent:
    def __init__(self, node):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = TaskPlanningPolicyNetwork().to(self.device)
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=3e-4)
        self.replay_buffer = TaskPlanningReplayBuffer()


    def infer_action(self, state: dict, deterministic=True):
        self.policy.eval()
        with torch.no_grad():
            rgb_image = torch.from_numpy(state["rgb_image"]).unsqueeze(0).to(self.device)
            bbox_list = torch.tensor(state["bbox_list"], 
                                     dtype=torch.float32,
                                     device=self.device).unsqueeze(0)
            class_list = torch.tensor(state["class_list"],
                                     dtype=torch.float32,
                                     device=self.device).unsqueeze(0)
            robot_state = torch.tensor(state["robot_state"]).unsqueeze(0).to(self.device)
            action = self.policy.forward(rgb_image, bbox_list, class_list, robot_state)
            action_numpy = action.cpu().numpy()
            print(f"Inferred action: {action_numpy}", flush=True)
            prob_sum = np.sum(action_numpy[:, :NUM_HEADS])
            if prob_sum > 0:
                action_numpy[:, :NUM_HEADS] /= prob_sum
            else:
                action_numpy[:, :NUM_HEADS] = np.ones(NUM_HEADS) / NUM_HEADS
        return action_numpy
    
    
    def update(self, bc_samples, past_bc_samples, val_samples):
        # Merge all samples into one batch for policy update
        if bc_samples is None and past_bc_samples is None:
            return
        
        if bc_samples is not None:
            rgb_image, bbox_list, class_list, robot_state, action, reward, done = bc_samples
            if past_bc_samples is not None:
                past_rgb_image, past_bbox_list, past_class_list, past_robot_state, past_action, past_reward, past_done = past_bc_samples
                rgb_image = torch.cat((rgb_image, past_rgb_image), dim=0)
                bbox_list = torch.cat((bbox_list, past_bbox_list), dim=0)
                class_list = torch.cat((class_list, past_class_list), dim=0)
                robot_state = torch.cat((robot_state, past_robot_state), dim=0)
                action = torch.cat((action, past_action), dim=0)
                reward = torch.cat((reward, past_reward), dim=0)
                done = torch.cat((done, past_done), dim=0)
        elif past_bc_samples is not None:
            rgb_image, bbox_list, class_list, robot_state, action, reward, done = past_bc_samples
            if bc_samples is not None:
                bc_rgb_image, bc_bbox_list, bc_class_list, bc_robot_state, bc_action, bc_reward, bc_done = bc_samples
                rgb_image = torch.cat((rgb_image, bc_rgb_image), dim=0)
                bbox_list = torch.cat((bbox_list, bc_bbox_list), dim=0)
                class_list = torch.cat((class_list, bc_class_list), dim=0)
                robot_state = torch.cat((robot_state, bc_robot_state), dim=0)
                action = torch.cat((action, bc_action), dim=0)
                reward = torch.cat((reward, bc_reward), dim=0)
                done = torch.cat((done, bc_done), dim=0)
        # Policy update using behavior cloning loss
        pred_action = self.policy.forward(rgb_image, bbox_list, class_list, robot_state)
        # Cross-entropy loss for bbox selection, MSE loss for skill selection
        bbox_loss = F.cross_entropy(pred_action[:, :NUM_HEADS], torch.argmax(action[:, :NUM_HEADS], dim=-1))
        skill_loss = F.mse_loss(pred_action[:, NUM_HEADS:], action[:, NUM_HEADS:])
        total_loss = bbox_loss + skill_loss
        self.policy_optimizer.zero_grad()
        total_loss.backward()
        self.policy_optimizer.step()
        
        print(f"Policy update - Total Loss: {total_loss.item():.4f}, BBox Loss: {bbox_loss.item():.4f}, Skill Loss: {skill_loss.item():.4f}", flush=True)
        
        return {
            "total_loss": total_loss.item(),
            "bbox_loss": bbox_loss.item(),
            "skill_loss": skill_loss.item()
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
