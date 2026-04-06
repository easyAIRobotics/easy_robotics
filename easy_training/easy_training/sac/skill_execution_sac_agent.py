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
# MAX_RANGE = 1.0

torch.autograd.set_detect_anomaly(True)

class SceneImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, kernel_size=7, stride=2, padding=3)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)

        self.fully_connected = nn.Linear(32 * 4 * 5, 64)

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

        self.fully_connected = nn.Linear(32 * 2 * 2, 64)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fully_connected(x))
        return x
        
        
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Bernoulli


class SkillExecutionPolicyNetwork(nn.Module):
    def __init__(self, action_dim, hidden_dim=128, gru_hidden=128, device="cuda"):
        super().__init__()

        self.scene_encoder = SceneImageEncoder()
        self.target_encoder = TargetImageEncoder()
        self.device = device
        state_dim = 64 + 64 + 3 + 15 + 1

        # -----------------------
        # GRU replaces FC stack
        # -----------------------
        self.gru = nn.GRU(
            input_size=state_dim,
            hidden_size=gru_hidden,
            num_layers=2,
            batch_first=True
        )
        self.gru_norm = nn.LayerNorm(gru_hidden)

        # single FC after GRU
        self.fc = nn.Linear(gru_hidden, hidden_dim)

        # heads
        self.mean = nn.Linear(hidden_dim, action_dim - 1)
        self.log_std = nn.Linear(hidden_dim, action_dim - 1)
        self.logit = nn.Linear(hidden_dim, 1)

        self.LOG_STD_MIN = -10
        self.LOG_STD_MAX = 0
        
    def flatten_parameters(self):
        self.gru.flatten_parameters()

    # ======================================================
    # Encode per timestep
    # ======================================================
    def encode_state(self, img, target_img, skill, robot):
        scene_z = self.scene_encoder(img)
        target_z = self.target_encoder(target_img)
        return torch.cat([target_z, skill, robot, scene_z], dim=-1)

    # ======================================================
    # Forward (sequence)
    # ======================================================
    def forward(self, img, target_img, skill, robot, hidden=None):
        """
        img: (B, T, C, H, W)
        """
        B, T = img.shape[:2]
        if hidden is None:
            hidden = torch.zeros(
                self.gru.num_layers,
                B,
                self.gru.hidden_size,
                device=self.device
            )
        # encode per timestep
        x = self.encode_state(
            img.reshape(B * T, *img.shape[2:]),
            target_img.reshape(B * T, *target_img.shape[2:]),
            skill.reshape(B * T, -1),
            robot.reshape(B * T, -1),
        )
        x = x.view(B, T, -1)
        # GRU
        out, hidden = self.gru(x, hidden)  # (B, T, H)
        out = self.gru_norm(out)
        h = out[:, -1, :]  # last timestep
        h = F.relu(self.fc(h) + h)
        mean = self.mean(h)
        log_std = torch.clamp(self.log_std(h), self.LOG_STD_MIN, self.LOG_STD_MAX)
        logit = self.logit(h)
        return mean, log_std, logit, hidden

    # ======================================================
    # Stochastic sampling
    # ======================================================
    def sample(self, img, target_img, skill, robot, hidden=None):
        mean, log_std, logit, hidden = self.forward(img, target_img, skill, robot, hidden)
        std = log_std.exp()
        normal = Normal(mean, std)

        z = normal.rsample()
        action_cont = torch.tanh(z)

        log_prob_cont = normal.log_prob(z)
        log_prob_cont -= torch.log(1 - action_cont.pow(2) + 1e-6)
        log_prob_cont = log_prob_cont.sum(dim=-1, keepdim=True)

        bern = Bernoulli(logits=logit)
        action_bin = bern.sample()

        log_prob_bin = bern.log_prob(action_bin)

        action = torch.cat([action_cont, action_bin], dim=-1)
        log_prob = log_prob_cont + log_prob_bin

        return action, log_prob, hidden

    # ======================================================
    # Deterministic
    # ======================================================
    def deterministic(self, img, target_img, skill, robot, hidden=None):
        mean, _, logit, hidden = self.forward(img, target_img, skill, robot, hidden)

        action_cont = torch.tanh(mean)
        action_bin = (torch.sigmoid(logit) > 0.5).float()

        return torch.cat([action_cont, action_bin], dim=-1), logit, hidden
    

class SkillExecutionCriticNetwork(nn.Module):
    def __init__(self, action_dim, hidden_dim=128, gru_hidden=128, device="cuda"):
        super().__init__()

        self.scene_encoder = SceneImageEncoder()
        self.target_encoder = TargetImageEncoder()
        self.device = device
        state_dim = 64 + 64 + 3 + 15 + 1

        # GRU only processes STATE
        self.gru = nn.GRU(
            input_size=state_dim,
            hidden_size=gru_hidden,
            num_layers=2,
            batch_first=True
        )
        self.gru_norm = nn.LayerNorm(gru_hidden)

        self.fc = nn.Linear(gru_hidden + action_dim, hidden_dim)

        # Twin Q heads
        self.q1 = nn.Linear(hidden_dim, 1)
        self.q2 = nn.Linear(hidden_dim, 1)
        
    def flatten_parameters(self):
        self.gru.flatten_parameters()

    def encode_state(self, img, target_img, skill, robot):
        scene_z = self.scene_encoder(img)
        target_z = self.target_encoder(target_img)
        return torch.cat([target_z, skill, robot, scene_z], dim=-1)

    def forward(self, img, target_img, skill, robot, action, hidden=None):
        B, T = img.shape[:2]

        # -----------------------
        # Encode state sequence
        # -----------------------
        state = self.encode_state(
            img.reshape(B * T, *img.shape[2:]),
            target_img.reshape(B * T, *target_img.shape[2:]),
            skill.reshape(B * T, -1),
            robot.reshape(B * T, -1),
        )

        state = state.view(B, T, -1)

        # -----------------------
        # GRU over STATE ONLY
        # -----------------------
        out, hidden = self.gru(state, hidden)
        out = self.gru_norm(out)

        h = out[:, -1, :]  # last timestep

        # -----------------------
        # Inject ACTION AFTER GRU
        # -----------------------
        action = action.view(B, -1)

        h = torch.cat([h, action], dim=-1)

        h = F.relu(self.fc(h))

        q1 = self.q1(h)
        q2 = self.q2(h)

        return q1, q2
    
    
JOINT_WEIGHT = 5.0
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
        
        self.hidden = None
        
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
        
        self.policy.flatten_parameters()
        self.q.flatten_parameters()
        self.target_q.flatten_parameters()
        
        # Optimizers
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.q_optimizer = torch.optim.Adam(self.q.parameters(), lr=lr)
        
        
    def infer_action(self, state: dict, deterministic=True):
        # ======================================================
        # 1. Extract
        # ======================================================
        img = state.get("image", None)
        target_img = state.get("target_image", None)
        skill = state.get("skill", None)
        robot = state.get("robot_state", None)

        if any(x is None for x in [img, target_img, skill, robot]):
            return None

        # ======================================================
        # 2. To torch + device
        # ======================================================
        def to_tensor(x):
            return torch.from_numpy(x).float().to(self.device)

        img = to_tensor(img)
        target_img = to_tensor(target_img)
        skill = to_tensor(skill)
        robot = to_tensor(robot)

        # ======================================================
        # 3. Add batch dim
        # ======================================================
        img = img.unsqueeze(0)
        target_img = target_img.unsqueeze(0)
        skill = skill.unsqueeze(0)
        robot = robot.unsqueeze(0)

        # ======================================================
        # 4. Image format fix (HWC → BCHW)
        # ======================================================
        img = img.permute(0, 3, 1, 2).contiguous()
        target_img = target_img.permute(0, 3, 1, 2).contiguous()

        # ======================================================
        # 5. Policy inference
        # ======================================================
        with torch.no_grad():
            if deterministic:
                action, _, self.hidden = self.policy.deterministic(
                    img, target_img, skill, robot, self.hidden
                )
            else:
                action, log_prob, self.hidden = self.policy.sample(
                    img, target_img, skill, robot, self.hidden
                )

        action = action.squeeze(0).cpu().numpy()

        # ======================================================
        # 6. Debug log (raw output)
        # ======================================================
        print(f"[Policy] raw action: {action}", flush=True)

        # ======================================================
        # 7. Post-processing (action scaling)
        # ======================================================
        scaled_action = action.copy()

        if ACTION_MODE == "joint_positions":
            scaled_action[:6] *= self.max_range
        else:
            scaled_action[:3] *= self.max_range

        # ======================================================
        # 8. Final log
        # ======================================================
        print(f"[Policy] scaled action: {scaled_action}", flush=True)

        return scaled_action
    

    def update(self, rl_samples=None, bc_samples=None, val_samples=None):
        start_time = time.time()
        torch.cuda.set_device(0)

        if rl_samples is None and bc_samples is None:
            return None

        ########################################
        # -------- PRINT INFO ------------------
        ########################################
        if rl_samples is not None:
            print(f"[SAC] RL batch: {rl_samples[0][0].shape[0]}", flush=True)
        if bc_samples is not None:
            print(f"[SAC] BC batch: {bc_samples[0][0].shape[0]}", flush=True)

        ########################################
        # -------- MERGE BATCHES ---------------
        ########################################
        def unpack(samples, use_bc=False):
            if samples is None:
                return None

            (img, target_img, skill, robot), j_act, e_act, rewards, dones, \
            (n_img, n_target_img, n_skill, n_robot) = samples

            actions = j_act if ACTION_MODE == "joint_positions" else e_act

            if ACTION_MODE == "joint_positions":
                actions = actions.clone()
                actions[:, :6] /= self.max_range
            else:
                actions = actions.clone()
                actions[:, :3] /= self.max_range

            return {
                "img": img,
                "target_img": target_img,
                "skill": skill,
                "robot": robot,
                "actions": actions,
                "rewards": rewards,
                "dones": dones,
                "next_img": n_img,
                "next_target_img": n_target_img,
                "next_skill": n_skill,
                "next_robot": n_robot,
            }

        rl = unpack(rl_samples)
        bc = unpack(bc_samples, use_bc=True)
        def concat(field):
            if rl is not None and bc is not None:
                return torch.cat([x[field] for x in [rl, bc] if x is not None], dim=0)
            elif rl is not None:
                return rl[field]
            elif bc is not None:
                return bc[field]
            else:
                return None

        img = concat("img")
        target_img = concat("target_img")
        skill = concat("skill")
        robot = concat("robot")
        actions = concat("actions")
        rewards = concat("rewards")
        dones = concat("dones")

        next_img = concat("next_img")
        next_target_img = concat("next_target_img")
        next_skill = concat("next_skill")
        next_robot = concat("next_robot")

        ########################################
        # -------- CRITIC UPDATE ---------------
        ########################################
        with torch.no_grad():
            next_action, next_log_prob, _ = self.policy.sample(
                next_img, next_target_img, next_skill, next_robot
            )
            tq1, tq2 = self.target_q(
                next_img, next_target_img, next_skill, next_robot, next_action
            )
            target_v = torch.min(tq1, tq2) - self.alpha * next_log_prob
            q_target = rewards[:, -1] + (1 - dones[:, -1]) * self.gamma * target_v
        q1, q2 = self.q(img, target_img, skill, robot, actions[:, -1, :])
        q1_loss = F.mse_loss(q1, q_target)
        q2_loss = F.mse_loss(q2, q_target)
        critic_loss = q1_loss + q2_loss
        self.q_optimizer.zero_grad()
        critic_loss.backward()
        self.q_optimizer.step()

        ########################################
        # -------- POLICY UPDATE ---------------
        ########################################
        new_actions, log_prob, _ = self.policy.sample(
            img, target_img, skill, robot
        )

        with torch.no_grad():
            q1_new, q2_new = self.q(
                img, target_img, skill, robot, new_actions
            )

        q_min = torch.min(q1_new, q2_new)
        sac_loss = (self.alpha * log_prob - q_min).mean()

        ########################################
        # -------- BC LOSS ---------------------
        ########################################
        bc_loss = torch.tensor(0.0, device=img.device)

        if bc is not None:
            bc_actions = bc["actions"]

            bc_pred, logit, _ = self.policy.deterministic(
                bc["img"],
                bc["target_img"],
                bc["skill"],
                bc["robot"]
            )

            joint_loss = F.mse_loss(
                bc_pred[:, :self.gripper_act_id],
                bc_actions[:, -1, :self.gripper_act_id]
            )

            suction_loss = F.binary_cross_entropy_with_logits(
                logit.squeeze(-1),
                bc_actions[:, -1, self.gripper_act_id]
            )

            bc_loss = JOINT_WEIGHT * joint_loss + SUCTION_WEIGHT * suction_loss

        total_policy_loss = sac_loss + self.bc_weight * bc_loss

        self.policy_optimizer.zero_grad()
        total_policy_loss.backward()
        self.policy_optimizer.step()
        ########################################
        # -------- SOFT UPDATE -----------------
        ########################################
        if self.step % 10 == 0:
            for tp, p in zip(self.target_q.parameters(), self.q.parameters()):
                tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

        self.step += 1

        ########################################
        # -------- LOGGING ---------------------
        ########################################
        print(
            f"[SAC] done in {time.time()-start_time:.2f}s | "
            f"Q1 {q1_loss.item():.4f} | "
            f"Q2 {q2_loss.item():.4f} | "
            f"SAC {sac_loss.item():.4f} | "
            f"BC {bc_loss.item():.4f}",
            flush=True
        )

        ########################################
        # -------- VALIDATION ------------------
        ########################################
        if val_samples is not None:
            val_img, val_target_img, val_skill, val_robot = val_samples[0]
            val_actions = val_samples[1] if ACTION_MODE == "joint_positions" else val_samples[2]

            if ACTION_MODE == "joint_positions":
                val_actions[:, :6] /= self.max_range
            else:
                val_actions[:, :3] /= self.max_range

            with torch.no_grad():
                pred, logit, _ = self.policy.deterministic(
                    val_img, val_target_img, val_skill, val_robot
                )

                joint_loss = F.mse_loss(
                    pred[:, :self.gripper_act_id],
                    val_actions[:, -1, :self.gripper_act_id]
                )

                suction_loss = F.binary_cross_entropy_with_logits(
                    logit.squeeze(-1),
                    val_actions[:, -1, self.gripper_act_id]
                )

                val_loss = JOINT_WEIGHT * joint_loss + SUCTION_WEIGHT * suction_loss

                print(
                    f"[VAL] loss {val_loss.item():.4f} | "
                    f"joint {joint_loss.item():.4f} | "
                    f"suction {suction_loss.item():.4f}",
                    flush=True
                )

        ########################################
        # -------- RETURN ----------------------
        ########################################
        return {
            "q1_loss": q1_loss.item(),
            "q2_loss": q2_loss.item(),
            "sac_loss": sac_loss.item(),
            "bc_loss": bc_loss.item(),
            "val_loss": val_loss.item() if val_samples is not None else 0.0,
        }
