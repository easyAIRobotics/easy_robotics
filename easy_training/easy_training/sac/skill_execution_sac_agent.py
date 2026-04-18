import time

import torch
import torch.nn as nn
from torch.distributions import Normal, Bernoulli
import torch.nn.functional as F

import numpy as np

from easy_training.sac.sac_agent import SACAgent
from easy_training.utils import *

# MAX_STEP_DELTA = 0.02
# MAX_STEP_ANGLE = np.pi / 24

MAX_RANGE = np.pi
# MAX_RANGE = 1.5

torch.autograd.set_detect_anomaly(True)

class SceneImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, kernel_size=7, stride=2, padding=3)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)

        self.fully_connected = nn.Linear(32 * 4 * 5, 256)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fully_connected(x))
        return x

class TargetImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, kernel_size=7, stride=2, padding=3)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)

        self.fully_connected = nn.Linear(32 * 2 * 2, 256)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fully_connected(x))
        return x
        
        
class SkillExecutionPolicyNetwork(nn.Module):
    def __init__(self, action_dim, hidden_dim=1024):
        super().__init__()

        self.scene_encoder = SceneImageEncoder()
        self.target_encoder = TargetImageEncoder()
        self.original_target_encoder = TargetImageEncoder()
        state_dim = 256 + 256 + 256 + 3 + 15 + 1

        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        # self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        # self.fc5 = nn.Linear(hidden_dim, hidden_dim)

        # 6 continuous
        self.mean = nn.Linear(hidden_dim, action_dim - 1)
        self.log_std = nn.Linear(hidden_dim, action_dim - 1)

        # 1 binary
        self.logit = nn.Linear(hidden_dim, 1)

        self.LOG_STD_MIN = -10
        self.LOG_STD_MAX = -5

    def forward(self, img, target_img, original_target_img, skill, robot):
        scene_z = self.scene_encoder(img)
        target_z = self.target_encoder(target_img)
        original_target_z = self.original_target_encoder(original_target_img)
        state = torch.cat([target_z, original_target_z, skill, robot, scene_z], dim=-1)

        h1 = F.relu(self.fc1(state))
        h2 = F.relu(self.fc2(h1)) + h1
        h3 = F.relu(self.fc3(h2)) + h2
        # h4 = F.relu(self.fc4(h3)) + h3
        # h5 = F.relu(self.fc5(h4)) + h4

        mean = self.mean(h3)
        log_std = torch.clamp(self.log_std(h3), self.LOG_STD_MIN, self.LOG_STD_MAX)
        logit = self.logit(h3)

        return mean, log_std, logit

    def sample(self, img, target_img, original_target_img, skill, robot):
        mean, log_std, logit = self.forward(img, target_img, original_target_img, skill, robot)

        # ===== continuous act =====
        std = log_std.exp()
        normal = Normal(mean, std)

        z = normal.rsample()
        action_cont = torch.tanh(z)

        log_prob_cont = normal.log_prob(z)
        log_prob_cont -= torch.log(1 - action_cont.pow(2) + 1e-6)
        log_prob_cont = log_prob_cont.sum(dim=-1, keepdim=True)

        # ===== binary act (1 dim) =====
        bern = Bernoulli(logits=logit)

        action_bin = bern.sample()   # {0,1}

        log_prob_bin = bern.log_prob(action_bin)
        log_prob_bin = log_prob_bin.sum(dim=-1, keepdim=True)

        # ===== combine =====
        action = torch.cat([action_cont, action_bin], dim=-1)
        log_prob = log_prob_cont + log_prob_bin
        return action, log_prob

    def deterministic(self, img, target_img, original_target_img, skill, robot):
        mean, _, logit = self.forward(img, target_img, original_target_img, skill, robot)
        robot_act = torch.tanh(mean)
        gripper_act = (torch.sigmoid(logit) > 0.5).float()
        return torch.cat([robot_act, gripper_act], dim=-1), logit


class SkillExecutionCriticNetwork(nn.Module):
    def __init__(self, action_dim, hidden_dim=1024):
        super().__init__()

        self.scene_encoder = SceneImageEncoder()
        self.target_encoder = TargetImageEncoder()
        self.original_target_encoder = TargetImageEncoder()

        state_dim = 256 + 256 + 256 + 3 + 15 + 1

        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        # self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        # self.fc5 = nn.Linear(hidden_dim, hidden_dim)

        self.fc1_ = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2_ = nn.Linear(hidden_dim, hidden_dim)
        self.fc3_ = nn.Linear(hidden_dim, hidden_dim)
        # self.fc4_ = nn.Linear(hidden_dim, hidden_dim)
        # self.fc5_ = nn.Linear(hidden_dim, hidden_dim)

        self.q = nn.Linear(hidden_dim, 1)
        self.q_ = nn.Linear(hidden_dim, 1)


    def forward(self, img, target_img, original_target_img, skill, robot, action):
        scene_z = self.scene_encoder(img)
        target_z = self.target_encoder(target_img)
        original_target_z = self.original_target_encoder(original_target_img)
        state = torch.cat([target_z, original_target_z, skill, robot, scene_z], dim=-1)
        x = torch.cat([state, action], dim=-1)

        # Q1
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1)) + h1
        h3 = F.relu(self.fc3(h2)) + h2
        # h4 = F.relu(self.fc4(h3)) + h3
        # h5 = F.relu(self.fc5(h4)) + h4

        # Q2
        h1_ = F.relu(self.fc1_(x))
        h2_ = F.relu(self.fc2_(h1_)) + h1_
        h3_ = F.relu(self.fc3_(h2_)) + h2_
        # h4_ = F.relu(self.fc4_(h3_)) + h3_
        # h5_ = F.relu(self.fc5_(h4_)) + h4_

        return self.q(h3), self.q_(h3_)
    
    
JOINT_WEIGHT = 50.0
SUCTION_WEIGHT = 1.0
ACTION_MODE = "joint_positions"  # "eef_pose" or "joint_positions"
# ACTION_MODE = "eef_pose"
ACTION_DIM = 10 if ACTION_MODE == "eef_pose" else 7

import copy
from rclpy.node import Node

class SkillExecutionSACAgent(SACAgent):
    def __init__(
        self, node: Node,
        gamma=0.98,
        tau=0.02,
        alpha=0.0,
        bc_weight=10.0,
        lr=5e-4
    ):
        super().__init__(node, "skill_execution_sac_agent")
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.bc_weight = bc_weight
        self.step = 0
        
        if ACTION_MODE == "joint_positions":
            self.gripper_act_id = 6
        else:
            self.gripper_act_id = 9
        
        # Delta scale
        # self._node.declare_parameter("max_step_delta", MAX_STEP_DELTA)
        # self.delta_pos_max = self._node.get_parameter("max_step_delta").get_parameter_value().double_value
        # self._node.declare_parameter("max_step_angle", MAX_STEP_ANGLE)
        # self.max_step_angle = self._node.get_parameter("max_step_angle").get_parameter_value().double_value
        
        self._node.declare_parameter("max_range", MAX_RANGE)
        self.max_range = self._node.get_parameter("max_range").get_parameter_value().double_value

        # Networks
        self.policy = SkillExecutionPolicyNetwork(ACTION_DIM).to(self.device)
        self.q = SkillExecutionCriticNetwork(ACTION_DIM).to(self.device)

        self.target_q = copy.deepcopy(self.q)
        
        # Optimizers
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.q_optimizer = torch.optim.Adam(self.q.parameters(), lr=lr)
        
        
    def infer_action(self, state: dict, deterministic=True):
        rl_img, rl_target_img, rl_original_target_img, obs_skill, obs_robot = state["image"], state["target_image"], state["original_target_image"], state["skill"], state["robot_state"]
        
        if rl_img is None or rl_target_img is None or rl_original_target_img is None or obs_skill is None or obs_robot is None:
            return None

        rl_img = torch.from_numpy(rl_img).float().to(self.device)
        rl_target_img = torch.from_numpy(rl_target_img).float().to(self.device)
        rl_original_target_img = torch.from_numpy(rl_original_target_img).float().to(self.device)
        obs_skill = torch.from_numpy(obs_skill).float().to(self.device)
        obs_robot = torch.from_numpy(obs_robot).float().to(self.device)

        # add batch dimension
        rl_img = rl_img.unsqueeze(0)
        rl_target_img = rl_target_img.unsqueeze(0)
        rl_original_target_img = rl_original_target_img.unsqueeze(0)
        obs_skill = obs_skill.unsqueeze(0)
        obs_robot = obs_robot.unsqueeze(0)

        # convert HWC → BCHW
        rl_img = rl_img.permute(0, 3, 1, 2)
        rl_target_img = rl_target_img.permute(0, 3, 1, 2)
        rl_original_target_img = rl_original_target_img.permute(0, 3, 1, 2)
        
        with torch.no_grad():
            if deterministic:
                action, _ = self.policy.deterministic(rl_img, rl_target_img, rl_original_target_img, obs_skill, obs_robot)
            else:
                action, log_prob = self.policy.sample(rl_img, rl_target_img, rl_original_target_img, obs_skill, obs_robot)
                print(f"[SkillExecutionSACAgent] Action log probability: {log_prob.item():.4f}", flush=True)
        action = action.cpu().numpy()
        
        print(f"[SkillExecutionSACAgent] Raw inferred action (delta_pos, quat, suction): {action}", flush=True)

        # -----------------------------
        # Post-processing
        # -----------------------------

        if ACTION_MODE == "joint_positions":
            action[..., :6] *= self.max_range
        else:
            action[..., :3] *= self.max_range

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
        target_imgs = []
        original_target_imgs = []
        skills = []
        robots = []
        actions_list = []
        rewards_list = []
        dones_list = []
        next_imgs = []
        next_target_imgs = []
        next_original_target_imgs = []
        next_skills = []
        next_robots = []

        # ---- RL ----
        if rl_samples is not None:
            (rl_img, rl_target_img, rl_original_target_img, rl_skill, rl_robot), rl_j_actions, rl_e_actions, rl_rewards, rl_dones, \
            (rl_next_img, rl_next_target_img, rl_next_original_target_img, rl_next_skill, rl_next_robot) = rl_samples

            imgs.append(rl_img)
            target_imgs.append(rl_target_img)
            original_target_imgs.append(rl_original_target_img)
            skills.append(rl_skill)
            robots.append(rl_robot)
            if ACTION_MODE == "joint_positions":
                actions_list.append(rl_j_actions)
            else:
                actions_list.append(rl_e_actions)
            rewards_list.append(rl_rewards)
            dones_list.append(rl_dones)

            next_imgs.append(rl_next_img)
            next_target_imgs.append(rl_next_target_img)
            next_original_target_imgs.append(rl_next_original_target_img)
            next_skills.append(rl_next_skill)
            next_robots.append(rl_next_robot)

        # ---- BC ----
        if bc_samples is not None:
            (bc_img, bc_target_img, bc_original_target_img, bc_skill, bc_robot), bc_j_actions, bc_e_actions, bc_rewards, bc_dones, \
            (bc_next_img, bc_next_target_img, bc_next_original_target_img, bc_next_skill, bc_next_robot) = bc_samples

            imgs.append(bc_img)
            target_imgs.append(bc_target_img)
            original_target_imgs.append(bc_original_target_img)
            skills.append(bc_skill)
            robots.append(bc_robot)
            if ACTION_MODE == "joint_positions":
                actions_list.append(bc_j_actions)
            else:
                actions_list.append(bc_e_actions)
            rewards_list.append(bc_rewards)
            dones_list.append(bc_dones)

            next_imgs.append(bc_next_img)
            next_target_imgs.append(bc_next_target_img)
            next_original_target_imgs.append(bc_next_original_target_img)
            next_skills.append(bc_next_skill)
            next_robots.append(bc_next_robot)

        # Concatenate only what exists
        img = torch.cat(imgs, dim=0)
        target_img = torch.cat(target_imgs, dim=0)
        original_target_img = torch.cat(original_target_imgs, dim=0)
        skill = torch.cat(skills, dim=0)
        robot = torch.cat(robots, dim=0)
        actions = torch.cat(actions_list, dim=0)
        if ACTION_MODE == "joint_positions":
            actions[:, :6] = actions[:, :6] / self.max_range
        else:
            actions[:, :3] = actions[:, :3] / self.max_range
        rewards = torch.cat(rewards_list, dim=0)
        dones = torch.cat(dones_list, dim=0)

        next_img = torch.cat(next_imgs, dim=0)
        next_target_img = torch.cat(next_target_imgs, dim=0)
        next_original_target_img = torch.cat(next_original_target_imgs, dim=0)
        next_skill = torch.cat(next_skills, dim=0)
        next_robot = torch.cat(next_robots, dim=0)

        ########################################
        # -------- Critic Update ---------------
        ########################################

        with torch.no_grad():
            next_action, next_log_prob = self.policy.sample(next_img, next_target_img, next_original_target_img, next_skill, next_robot)
            target_q1, target_q2 = self.target_q(next_img, next_target_img, next_original_target_img, next_skill, next_robot, next_action)
            target_v = torch.min(target_q1, target_q2) - self.alpha * next_log_prob
            q_target = rewards + (1 - dones) * self.gamma * target_v

        q1_pred, q2_pred = self.q(img, target_img, original_target_img, skill, robot, actions)

        q1_loss = F.mse_loss(q1_pred, q_target)
        q2_loss = F.mse_loss(q2_pred, q_target)
        critic_loss = q1_loss + q2_loss

        self.q_optimizer.zero_grad()
        critic_loss.backward()
        self.q_optimizer.step()

        ########################################
        # -------- Policy Update ---------------
        ########################################
        new_actions, log_prob = self.policy.sample(img, target_img, original_target_img, skill, robot)
        with torch.no_grad():
            q1_new, q2_new = self.q(img, target_img, original_target_img, skill, robot, new_actions)
        q_min = torch.min(q1_new, q2_new)

        sac_loss = (self.alpha * log_prob - q_min).mean()

        # ----- BC imitation loss (only if BC exists) -----
        if bc_samples is not None:
            bc_batch_size = bc_img.shape[0]

            bc_actions = actions[-bc_batch_size:]
            bc_actions_pred, logit = self.policy.deterministic(bc_img, bc_target_img, bc_original_target_img, bc_skill, bc_robot)
            
            bc_joint_loss = F.mse_loss(bc_actions_pred[:, :self.gripper_act_id], bc_actions[:, :self.gripper_act_id])
            suction_loss = F.binary_cross_entropy_with_logits(logit.squeeze(-1), bc_actions[:, self.gripper_act_id])
            
            bc_loss = JOINT_WEIGHT * bc_joint_loss + SUCTION_WEIGHT * suction_loss

        else:
            bc_loss = 0.0

        total_policy_loss = sac_loss + self.bc_weight * bc_loss
        total_policy_loss = self.bc_weight * bc_loss

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
        ########################################
        # -------- Sample Loss -----------------
        ########################################
        
        if val_samples is not None:
            val_img, val_target_img, val_original_target_img, val_skill, val_robot = val_samples[0]
            if ACTION_MODE == "joint_positions":
                val_actions = val_samples[1]
                val_actions[:, :6] = val_actions[:, :6] / self.max_range
            else:
                val_actions = val_samples[2]
                val_actions[:, :3] = val_actions[:, :3] / self.max_range

            with torch.no_grad():
                val_actions_pred, logit = self.policy.deterministic(val_img, val_target_img, val_original_target_img, val_skill, val_robot)
                val_joint_loss = F.mse_loss(val_actions_pred[:, :self.gripper_act_id], val_actions[:, :self.gripper_act_id])
                suction_loss = F.binary_cross_entropy_with_logits(logit.squeeze(-1), val_actions[:, self.gripper_act_id])
                sample_loss = JOINT_WEIGHT * val_joint_loss + SUCTION_WEIGHT * suction_loss
                print(f"[SkillExecutionSACAgent] Validation loss: {sample_loss.item():.4f}, joint_loss: {val_joint_loss.item():.4f}, suction_loss: {suction_loss.item():.4f}", flush=True)
                print(f"[SkillExecutionSACAgent] Val joint states: {val_robot.cpu().detach().numpy()[:2, 9:15]}, skill vec: {val_skill[:2].cpu().detach().numpy()}", flush=True)
                print(f"[SkillExecutionSACAgent] Val actions: {val_actions[:2].cpu().detach().numpy() * self.max_range} \n versus predict: {val_actions_pred[:2].cpu().detach().numpy() * self.max_range}", flush=True)

        return {
            "q1_loss": q1_loss.item(),
            "q2_loss": q2_loss.item(),
            "sac_loss": sac_loss.item(),
            "bc_loss": bc_joint_loss.item() if bc_samples is not None else 0.0,
            "val_loss": val_joint_loss.item() if val_samples is not None else 0.0,
        }
        