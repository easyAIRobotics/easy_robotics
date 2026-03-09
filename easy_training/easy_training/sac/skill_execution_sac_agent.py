import time

import torch
import torch.nn as nn
from torch.distributions import Normal
import torch.nn.functional as F

import numpy as np

from easy_training.sac.sac_agent import SACAgent

MAX_STEP_DELTA = 0.02

torch.autograd.set_detect_anomaly(True)

class MaskedPointCloudEncoder(nn.Module):
    """
    Input: (B, 4, 120, 160) where 4 channels are (x, y, z, mask)
    Output: (B, 1024) latent vector
    """
    
    def __init__(self):
        super(MaskedPointCloudEncoder, self).__init__()
        
        # 120x160 → 60x80
        self.block1 = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=3, padding=1),
            nn.GroupNorm(8, 32),
            nn.ReLU(),
            nn.AvgPool2d(2)
        )

        # 60x80 → 30x40
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.GroupNorm(8, 64),
            nn.ReLU(),
            nn.AvgPool2d(2)
        )

        # 30x40 → 15x20
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.GroupNorm(8, 128),
            nn.ReLU(),
            nn.AvgPool2d(2)
        )

        # 15x20 → 7x10
        self.block4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.GroupNorm(16, 256),
            nn.ReLU(),
            nn.AvgPool2d(2)
        )

        # 7x10 → 3x5
        self.block5 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.GroupNorm(16, 512),
            nn.ReLU(),
            nn.AvgPool2d(2)
        )

        # Fully connected layer
        self.fc = nn.Linear(512 * 3 * 5, 1024)
        
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)

        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x
        
class SkillExecutionPolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)

        self.LOG_STD_MIN = -10
        self.LOG_STD_MAX = 1

    def forward(self, state):
        h = self.net(state)

        mean = self.mean(h)
        log_std = torch.clamp(
            self.log_std(h),
            self.LOG_STD_MIN,
            self.LOG_STD_MAX
        )

        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()

        normal = Normal(mean, std)
        z = normal.rsample()

        # SAC action
        action = torch.tanh(z)

        # Tanh correction
        log_prob = normal.log_prob(z)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        return action, log_prob

    def deterministic(self, state):
        mean, _ = self.forward(state)
        action = torch.tanh(mean)
        return action
    
class SkillExecutionCriticNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()

        self.q = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, 1)
        )
        
        self._init_params()

    def _init_params(self):
        for m in self.modules():

            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

        # Small initialization for final Q layer
        nn.init.uniform_(self.q[-1].weight, -1e-3, 1e-3)
        nn.init.uniform_(self.q[-1].bias, -1e-3, 1e-3)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.q(x)

STATE_DIM = 1024 + 3 + 13 + 1  # 1024 from encoder + 3 for skill ID + 7 for current eef and joint positions + 1 for suction state
ACTION_DIM = 8

import copy
import os
from rclpy.node import Node

class SkillExecutionSACAgent(SACAgent):
    def __init__(
        self, node: Node,
        gamma=0.99,
        tau=0.005,
        alpha=0.1,
        bc_weight=10.0,
        lr=3e-4
    ):
        super().__init__(node, "skill_execution_sac_agent")
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.bc_weight = bc_weight
        
        # Delta scale
        self._node.declare_parameter("max_step_delta", MAX_STEP_DELTA)
        self.delta_pos_max = self._node.get_parameter("max_step_delta").get_parameter_value().double_value

        # Networks
        self.encoder = MaskedPointCloudEncoder().to(self.device)
        self.policy = SkillExecutionPolicyNetwork(STATE_DIM, ACTION_DIM).to(self.device)
        self.q1 = SkillExecutionCriticNetwork(STATE_DIM, ACTION_DIM).to(self.device)
        self.q2 = SkillExecutionCriticNetwork(STATE_DIM, ACTION_DIM).to(self.device)

        self.target_q1 = copy.deepcopy(self.q1)
        self.target_q2 = copy.deepcopy(self.q2)
        
        # Optimizers
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.q1_optimizer = torch.optim.Adam(self.q1.parameters(), lr=lr)
        self.q2_optimizer = torch.optim.Adam(self.q2.parameters(), lr=lr)
        self.encoder_optimizer = torch.optim.Adam(self.encoder.parameters(), lr=lr)
        
        
    def infer_action(self, state: dict, deterministic=True):
        rl_img, obs_skill, obs_robot = state["image"], state["skill"], state["robot_state"]

        rl_img = torch.from_numpy(rl_img).float().to(self.device)
        obs_skill = torch.from_numpy(obs_skill).float().to(self.device)
        obs_robot = torch.from_numpy(obs_robot).float().to(self.device)

        # add batch dimension
        rl_img = rl_img.unsqueeze(0)
        obs_skill = obs_skill.unsqueeze(0)
        obs_robot = obs_robot.unsqueeze(0)

        # convert HWC → BCHW
        rl_img = rl_img.permute(0, 3, 1, 2)
        
        z = self.encoder(rl_img)
        state_vec = torch.cat([z, obs_skill, obs_robot], dim=-1)

        with torch.no_grad():
            if deterministic:
                action = self.policy.deterministic(state_vec)
            else:
                action, _ = self.policy.sample(state_vec)

        action = action.cpu().numpy()

        # -----------------------------
        # Post-processing
        # -----------------------------

        # delta position scaling
        delta_pos = action[..., :3] * self.delta_pos_max

        # quaternion normalization
        quat = action[..., 3:7]
        norm = np.linalg.norm(quat, axis=-1, keepdims=True) + 1e-8
        quat = quat / norm

        # enforce positive w
        mask = quat[..., 3:4] < 0
        quat = np.where(mask, -quat, quat)

        # suction command
        suction = action[..., 7:8]
        suction = np.clip(suction, -1, 1)

        processed_action = np.concatenate(
            [delta_pos, quat, suction],
            axis=-1
        )

        return processed_action
    

    def update(self, rl_samples=None, bc_samples=None):
        start_update_time = time.time()
        torch.cuda.set_device(0)
        if rl_samples is None and bc_samples is None:
            print("[SkillExecutionSACAgent] No samples available. Skipping.")
            return None
        
        if rl_samples is not None:
            print(f"[SkillExecutionSACAgent] Updating with RL samples. Batch size: {rl_samples[0][0].shape[0]}", flush=True)
        if bc_samples is not None:
            print(f"[SkillExecutionSACAgent] Updating with BC samples. Batch size: {bc_samples[0][0].shape[0]}", flush=True)

        ########################################
        # --------- Build Combined Batch ------
        ########################################

        imgs = []
        skills = []
        robots = []
        actions_list = []
        rewards_list = []
        next_imgs = []
        next_skills = []
        next_robots = []

        # ---- RL ----
        if rl_samples is not None:
            (rl_img, rl_skill, rl_robot), rl_actions, rl_rewards, \
            (rl_next_img, rl_next_skill, rl_next_robot) = rl_samples

            imgs.append(rl_img)
            skills.append(rl_skill)
            robots.append(rl_robot)
            actions_list.append(rl_actions)
            rewards_list.append(rl_rewards)

            next_imgs.append(rl_next_img)
            next_skills.append(rl_next_skill)
            next_robots.append(rl_next_robot)

        # ---- BC ----
        if bc_samples is not None:
            (bc_img, bc_skill, bc_robot), bc_actions, bc_rewards, \
            (bc_next_img, bc_next_skill, bc_next_robot) = bc_samples

            imgs.append(bc_img)
            skills.append(bc_skill)
            robots.append(bc_robot)
            actions_list.append(bc_actions)
            rewards_list.append(bc_rewards)

            next_imgs.append(bc_next_img)
            next_skills.append(bc_next_skill)
            next_robots.append(bc_next_robot)

        # Concatenate only what exists
        img = torch.cat(imgs, dim=0)
        skill = torch.cat(skills, dim=0)
        robot = torch.cat(robots, dim=0)
        actions = torch.cat(actions_list, dim=0)
        actions[:, :3] = torch.clamp(actions[:, :3] / self.delta_pos_max, -1.0, 1.0)
        rewards = torch.cat(rewards_list, dim=0)

        next_img = torch.cat(next_imgs, dim=0)
        next_skill = torch.cat(next_skills, dim=0)
        next_robot = torch.cat(next_robots, dim=0)

        ########################################
        # -------- Encode ----------------------
        ########################################

        z = self.encoder(img)
        next_z = self.encoder(next_img)

        state = torch.cat([z, skill, robot], dim=-1)
        next_state = torch.cat([next_z, next_skill, next_robot], dim=-1)

        ########################################
        # -------- Critic Update ---------------
        ########################################

        with torch.no_grad():
            next_action, next_log_prob = self.policy.sample(next_state)
            target_q1 = self.target_q1(next_state, next_action)
            target_q2 = self.target_q2(next_state, next_action)
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_prob
            q_target = rewards + self.gamma * target_q

        q1_pred = self.q1(state, actions)
        q2_pred = self.q2(state, actions)

        q1_loss = F.mse_loss(q1_pred, q_target)
        q2_loss = F.mse_loss(q2_pred, q_target)
        critic_loss = q1_loss + q2_loss

        self.q1_optimizer.zero_grad()
        self.q2_optimizer.zero_grad()
        self.encoder_optimizer.zero_grad()
        critic_loss.backward()
        self.q1_optimizer.step()
        self.q2_optimizer.step()
        self.encoder_optimizer.step()

        ########################################
        # -------- Policy Update ---------------
        ########################################
        z = self.encoder(img)
        policy_state = torch.cat([z, skill, robot], dim=-1)
        new_actions, log_prob = self.policy.sample(policy_state)
        q1_new = self.q1(policy_state, new_actions)
        q2_new = self.q2(policy_state, new_actions)
        q_min = torch.min(q1_new, q2_new)

        sac_loss = (self.alpha * log_prob - q_min).mean()

        # ----- BC imitation loss (only if BC exists) -----
        if bc_samples is not None:
            bc_batch_size = bc_img.shape[0]
            bc_state = policy_state[-bc_batch_size:]  # last part belongs to BC
            bc_actions = actions[-bc_batch_size:]
            bc_actions_pred, _ = self.policy.sample(bc_state)
            bc_loss = F.mse_loss(bc_actions_pred, bc_actions)
        else:
            bc_loss = 0.0

        total_policy_loss = sac_loss + self.bc_weight * bc_loss

        self.policy_optimizer.zero_grad()
        total_policy_loss.backward()
        self.policy_optimizer.step()

        ########################################
        # -------- Soft Update -----------------
        ########################################

        for target_param, param in zip(self.target_q1.parameters(), self.q1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for target_param, param in zip(self.target_q2.parameters(), self.q2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
        print(f"[SkillExecutionSACAgent] Update complete. Time taken: {time.time() - start_update_time:.2f} seconds. Q1 Loss: {q1_loss.item():.4f}, Q2 Loss: {q2_loss.item():.4f}, SAC Loss: {sac_loss.item():.4f}, BC Loss: {bc_loss.item() if bc_samples is not None else 0.0:.4f}", flush=True)
        print(f"[SkillExecutionSACAgent] Update complete. Q1 Loss: {q1_loss.item():.4f}, Q2 Loss: {q2_loss.item():.4f}, SAC Loss: {sac_loss.item():.4f}, BC Loss: {bc_loss.item() if bc_samples is not None else 0.0:.4f}", flush=True)

        return {
            "q1_loss": q1_loss.item(),
            "q2_loss": q2_loss.item(),
            "sac_loss": sac_loss.item(),
            "bc_loss": bc_loss.item() if bc_samples is not None else 0.0
        }
        