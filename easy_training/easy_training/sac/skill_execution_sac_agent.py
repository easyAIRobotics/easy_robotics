import time

import torch
import torch.nn as nn
from torch.distributions import Normal
import torch.nn.functional as F

import numpy as np

from easy_training.sac.sac_agent import SACAgent
from easy_training.utils import *

# MAX_STEP_DELTA = 0.02
# MAX_STEP_ANGLE = np.pi / 24

MAX_RANGE = np.pi

torch.autograd.set_detect_anomaly(True)

class SpatialSoftmax(nn.Module):
    def __init__(self, height, width, channel):
        super().__init__()
        self.height = height
        self.width = width
        self.channel = channel

        pos_x, pos_y = torch.meshgrid(
            torch.linspace(-1, 1, width),
            torch.linspace(-1, 1, height),
            indexing='xy'
        )
        self.register_buffer("pos_x", pos_x.reshape(-1))
        self.register_buffer("pos_y", pos_y.reshape(-1))

    def forward(self, feature):
        B, C, H, W = feature.shape

        feature = feature.view(B, C, H * W)
        softmax_attention = F.softmax(feature, dim=-1)

        expected_x = torch.sum(self.pos_x * softmax_attention, dim=-1)
        expected_y = torch.sum(self.pos_y * softmax_attention, dim=-1)

        keypoints = torch.cat([expected_x, expected_y], dim=-1)
        return keypoints  # (B, 2C)


class MaskAwareBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        # Main branch (geometry)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Mask branch (for gating)
        self.mask_conv = nn.Conv2d(1, out_channels, 3, stride, 1)

        # Skip connection
        if in_channels != out_channels or stride != 1:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.skip = nn.Identity()

        self.relu = nn.ReLU()

    def forward(self, x, mask):
        """
        x: (B, C, H, W)
        mask: (B, 1, H, W)
        """

        identity = self.skip(x)

        # ---- Main branch ----
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # ---- Mask gating ----
        gate = torch.sigmoid(self.mask_conv(mask))
        out = out * gate

        # ---- Residual add ----
        out = out + identity
        out = self.relu(out)

        # ---- Downsample mask ----
        mask = F.interpolate(mask, size=out.shape[-2:], mode='nearest')

        return out, mask


# ---------------------------
# Main Encoder
# ---------------------------
class MaskAwareEncoder(nn.Module):
    def __init__(self, latent_dim=128, use_spatial_softmax=True):
        super().__init__()

        self.use_spatial_softmax = use_spatial_softmax

        # Initial split
        self.input_conv = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)

        # Backbone
        self.block1 = MaskAwareBlock(32, 64, stride=2)   # 120x160 → 60x80
        self.block2 = MaskAwareBlock(64, 128, stride=2)  # → 30x40
        self.block3 = MaskAwareBlock(128, 256, stride=2) # → 15x20

        if use_spatial_softmax:
            self.spatial_softmax = SpatialSoftmax(15, 20, 256)
            self.fc = nn.Sequential(
                nn.Linear(256 * 2, 256),
                nn.ReLU(),
                nn.Linear(256, latent_dim)
            )
        else:
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Sequential(
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, latent_dim)
            )

    def forward(self, x):
        """
        x: (B, 4, 120, 160)
        channels: x,y,z,m
        """

        xyz = x[:, :3]        # (B,3,H,W)
        mask = x[:, 3:4]      # (B,1,H,W)

        # Apply mask early (important!)
        xyz = xyz * mask

        # Initial conv
        feat = self.input_conv(xyz)

        # Backbone
        feat, mask = self.block1(feat, mask)
        feat, mask = self.block2(feat, mask)
        feat, mask = self.block3(feat, mask)

        # Head
        if self.use_spatial_softmax:
            feat = self.spatial_softmax(feat)
        else:
            feat = self.pool(feat).view(feat.size(0), -1)

        latent = self.fc(feat)

        return latent
        
class SkillExecutionPolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()

        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)

        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)

        self.LOG_STD_MIN = -10
        self.LOG_STD_MAX = 0

    def forward(self, x):

        h1 = F.relu(self.fc1(x))

        h2 = F.relu(self.fc2(h1))
        h2 = h2 + h1          # residual

        h3 = F.relu(self.fc3(h2))
        h3 = h3 + h2          # residual

        mean = self.mean(h3)

        log_std = torch.clamp(
            self.log_std(h3),
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
        
        # log_prob = normal.log_prob(z).sum(dim=-1, keepdim=True)
        # return z, log_prob

    def deterministic(self, state):
        mean, _ = self.forward(state)
        return torch.tanh(mean)
    
class SkillExecutionCriticNetwork(nn.Module):

    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()

        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        
        
        self.fc1_ = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2_ = nn.Linear(hidden_dim, hidden_dim)
        self.fc3_ = nn.Linear(hidden_dim, hidden_dim)

        self.q = nn.Linear(hidden_dim, 1)
        self.q_ = nn.Linear(hidden_dim, 1)

        self._init_params()

    def _init_params(self):
        for m in self.modules():

            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

        nn.init.uniform_(self.q.weight, -1e-3, 1e-3)
        nn.init.uniform_(self.q.bias, -1e-3, 1e-3)

    def forward(self, state, action):

        x = torch.cat([state, action], dim=-1)

        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        h2 = h2 + h1
        
        h3 = F.relu(self.fc3(h2))
        h3 = h3 + h2
        
        h1_ = F.relu(self.fc1_(x))
        h2_ = F.relu(self.fc2_(h1_))
        h2_ = h2_ + h1_
        
        h3_ = F.relu(self.fc3_(h2_))
        h3_ = h3_ + h2_

        return self.q(h3), self.q_(h3_)

STATE_DIM = 128 + 3 + 15 + 1  # 128 from encoder + 3 for skill ID + 15 for current eef and joint positions + 1 for suction state
ACTION_DIM = 7

import copy
from rclpy.node import Node

class SkillExecutionSACAgent(SACAgent):
    def __init__(
        self, node: Node,
        gamma=0.98,
        tau=0.02,
        alpha=0.02,
        bc_weight=10.0,
        lr=5e-4
    ):
        super().__init__(node, "skill_execution_sac_agent")
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.bc_weight = bc_weight
        self.step = 0
        
        # Delta scale
        # self._node.declare_parameter("max_step_delta", MAX_STEP_DELTA)
        # self.delta_pos_max = self._node.get_parameter("max_step_delta").get_parameter_value().double_value
        # self._node.declare_parameter("max_step_angle", MAX_STEP_ANGLE)
        # self.max_step_angle = self._node.get_parameter("max_step_angle").get_parameter_value().double_value
        
        self._node.declare_parameter("max_range", MAX_RANGE)
        self.max_range = self._node.get_parameter("max_range").get_parameter_value().double_value

        # Networks
        self.encoder = MaskAwareEncoder().to(self.device)
        self.policy = SkillExecutionPolicyNetwork(STATE_DIM, ACTION_DIM).to(self.device)
        self.q = SkillExecutionCriticNetwork(STATE_DIM, ACTION_DIM).to(self.device)

        self.target_q = copy.deepcopy(self.q)
        
        # Optimizers
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.q_optimizer = torch.optim.Adam(self.q.parameters(), lr=lr)
        self.encoder_optimizer = torch.optim.Adam(self.encoder.parameters(), lr=lr)
        
        
    def infer_action(self, state: dict, deterministic=True):
        rl_img, obs_skill, obs_robot = state["image"], state["skill"], state["robot_state"]
        
        if rl_img is None or obs_skill is None or obs_robot is None:
            return None

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
                action, log_prob = self.policy.sample(state_vec)
                print(f"[SkillExecutionSACAgent] Action log probability: {log_prob.item():.4f}", flush=True)
        action = action.cpu().numpy()
        
        print(f"[SkillExecutionSACAgent] Raw inferred action (delta_pos, quat, suction): {action}", flush=True)

        # -----------------------------
        # Post-processing
        # -----------------------------

        # delta position scaling
        action[..., :6] *= self.max_range

        # delta rotation scaling
        # delta_rot = action[..., 3:6] * self.delta_angle_max
        
        # processed_action = np.concatenate([action[..., :3], action[..., 3:9], action[..., 9:]], axis=-1)
        
        print(f"[SkillExecutionSACAgent] Inferred action (delta_pos, delta_rot, suction): {action}", flush=True)
        return action
    

    def update(self, rl_samples=None, bc_samples=None, val_samples=None):
        start_update_time = time.time()
        torch.cuda.set_device(0)
        if rl_samples is None and bc_samples is None:
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
        dones_list = []
        next_imgs = []
        next_skills = []
        next_robots = []

        # ---- RL ----
        if rl_samples is not None:
            (rl_img, rl_skill, rl_robot), rl_actions, rl_rewards, rl_dones, \
            (rl_next_img, rl_next_skill, rl_next_robot) = rl_samples

            imgs.append(rl_img)
            skills.append(rl_skill)
            robots.append(rl_robot)
            actions_list.append(rl_actions)
            rewards_list.append(rl_rewards)
            dones_list.append(rl_dones)

            next_imgs.append(rl_next_img)
            next_skills.append(rl_next_skill)
            next_robots.append(rl_next_robot)

        # ---- BC ----
        if bc_samples is not None:
            (bc_img, bc_skill, bc_robot), bc_actions, bc_rewards, bc_dones, \
            (bc_next_img, bc_next_skill, bc_next_robot) = bc_samples

            imgs.append(bc_img)
            skills.append(bc_skill)
            robots.append(bc_robot)
            actions_list.append(bc_actions)
            rewards_list.append(bc_rewards)
            dones_list.append(bc_dones)

            next_imgs.append(bc_next_img)
            next_skills.append(bc_next_skill)
            next_robots.append(bc_next_robot)

        # Concatenate only what exists
        img = torch.cat(imgs, dim=0)
        skill = torch.cat(skills, dim=0)
        robot = torch.cat(robots, dim=0)
        actions = torch.cat(actions_list, dim=0)
        actions[:, :6] = actions[:, :6] / self.max_range
        rewards = torch.cat(rewards_list, dim=0)
        dones = torch.cat(dones_list, dim=0)

        next_img = torch.cat(next_imgs, dim=0)
        next_skill = torch.cat(next_skills, dim=0)
        next_robot = torch.cat(next_robots, dim=0)

        ########################################
        # -------- Encode ----------------------
        ########################################

        z = self.encoder(img)
        with torch.no_grad():
            next_z = self.encoder(next_img)

        state = torch.cat([z, skill, robot], dim=-1)
        next_state = torch.cat([next_z, next_skill, next_robot], dim=-1)

        ########################################
        # -------- Critic Update ---------------
        ########################################

        with torch.no_grad():
            next_action, next_log_prob = self.policy.sample(next_state)
            target_q1, target_q2 = self.target_q(next_state, next_action)
            target_v = torch.min(target_q1, target_q2) - self.alpha * next_log_prob
            q_target = rewards + (1 - dones) * self.gamma * target_v

        q1_pred, q2_pred = self.q(state, actions)

        q1_loss = F.mse_loss(q1_pred, q_target)
        q2_loss = F.mse_loss(q2_pred, q_target)
        critic_loss = q1_loss + q2_loss

        self.q_optimizer.zero_grad()
        self.encoder_optimizer.zero_grad()
        critic_loss.backward()
        self.q_optimizer.step()
        self.encoder_optimizer.step()

        ########################################
        # -------- Policy Update ---------------
        ########################################
        policy_state = torch.cat([z.detach(), skill, robot], dim=-1)
        new_actions, log_prob = self.policy.sample(policy_state)
        q1_new, q2_new = self.q(policy_state, new_actions)
        q_min = torch.min(q1_new, q2_new)

        sac_loss = (self.alpha * log_prob - q_min).mean()

        # ----- BC imitation loss (only if BC exists) -----
        if bc_samples is not None:
            bc_batch_size = bc_img.shape[0]

            bc_state = policy_state[-bc_batch_size:]
            bc_actions = actions[-bc_batch_size:]
            deterministic_actions = self.policy.deterministic(bc_state)
            bc_actions_pred = deterministic_actions
            bc_loss = F.mse_loss(bc_actions_pred, bc_actions)

        else:
            bc_loss = 0.0

        total_policy_loss = sac_loss + self.bc_weight * bc_loss

        self.policy_optimizer.zero_grad()
        total_policy_loss.backward()
        self.policy_optimizer.step()
        
        # if bc_samples is not None:
            # print(f"[Robot eef pose] {bc_state[:5, -25:-16].cpu().detach().numpy()}", flush=True)
            # print(f"[SkillExecutionSACAgent] bc_actions: {bc_actions[:2].cpu().detach().numpy()} \n vesus predict: {bc_actions_pred[:2].cpu().detach().numpy()}", flush=True)

        ########################################
        # -------- Soft Update -----------------
        ########################################

        if self.step % 10 == 0:
            for target_param, param in zip(self.target_q.parameters(), self.q.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        self.step += 1
        print(f"[SkillExecutionSACAgent] Update complete. Time taken: {time.time() - start_update_time:.2f} seconds. Q1 Loss: {q1_loss.item():.4f}, Q2 Loss: {q2_loss.item():.4f}, SAC Loss: {sac_loss.item():.4f}, BC Loss: {bc_loss.item() if bc_samples is not None else 0.0:.4f}", flush=True)

        ########################################
        # -------- Sample Loss -----------------
        ########################################
        
        if val_samples is not None:
            val_img, val_skill, val_robot = val_samples[0]
            val_actions = val_samples[1]
            val_actions[:, :6] = val_actions[:, :6] / self.max_range

            with torch.no_grad():
                z = self.encoder(val_img)
                val_state = torch.cat([z, val_skill, val_robot], dim=-1)

                val_actions_pred = self.policy.deterministic(val_state)
                sample_loss = F.mse_loss(val_actions_pred, val_actions)
                
            print(f"[SkillExecutionSACAgent] Val actions: {val_actions[:2].cpu().detach().numpy()} \n versus predict: {val_actions_pred[:2].cpu().detach().numpy()}", flush=True)

        return {
            "q1_loss": q1_loss.item(),
            "q2_loss": q2_loss.item(),
            "sac_loss": sac_loss.item(),
            "bc_loss": bc_loss.item() if bc_samples is not None else 0.0,
            "val_loss": sample_loss.item() if val_samples is not None else 0.0,
        }
        