import torch
import numpy as np

class SkillExecutionReplayBuffer:
    def __init__(
        self,
        capacity,
        image_shape,        # (M, N, C)
        skill_dim,
        robot_state_dim,
        action_dim,
        device="cuda"
    ):
        self.capacity = capacity
        self.device = device
        self.ptr = 0
        self.size = 0

        M, N, C = image_shape

        # --------------------
        # Current observation
        # --------------------
        self.images = np.zeros((capacity, M, N, C), dtype=np.float32)
        self.skills = np.zeros((capacity, skill_dim), dtype=np.float32)
        self.robot_states = np.zeros((capacity, robot_state_dim), dtype=np.float32)

        # --------------------
        # Next observation
        # --------------------
        self.next_images = np.zeros((capacity, M, N, C), dtype=np.float32)
        self.next_skills = np.zeros((capacity, skill_dim), dtype=np.float32)
        self.next_robot_states = np.zeros((capacity, robot_state_dim), dtype=np.float32)

        # --------------------
        # RL signals
        # --------------------
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)

    # ======================================================
    # Add transition (works for RL and BC)
    # ======================================================
    def add(
        self,
        image,
        skill,
        robot_state,
        action,
        reward,
        next_image,
        next_skill,
        next_robot_state,
    ):
        self.images[self.ptr] = image
        self.skills[self.ptr] = skill
        self.robot_states[self.ptr] = robot_state

        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward

        self.next_images[self.ptr] = next_image
        self.next_skills[self.ptr] = next_skill
        self.next_robot_states[self.ptr] = next_robot_state

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    # ======================================================
    # Sample batch
    # ======================================================
    def sample(self, batch_size):
        if self.size == 0:
            print("[SkillExecutionReplayBuffer] Not enough samples to draw a batch. Returning None.")
            return None
            
        idx = np.random.randint(0, self.size, size=batch_size)

        # -------- Current --------
        images = torch.from_numpy(self.images[idx]).to(self.device)
        skills = torch.from_numpy(self.skills[idx]).to(self.device)
        robot_states = torch.from_numpy(self.robot_states[idx]).to(self.device)

        # -------- Next --------
        next_images = torch.from_numpy(self.next_images[idx]).to(self.device)
        next_skills = torch.from_numpy(self.next_skills[idx]).to(self.device)
        next_robot_states = torch.from_numpy(self.next_robot_states[idx]).to(self.device)

        actions = torch.from_numpy(self.actions[idx]).to(self.device)
        rewards = torch.from_numpy(self.rewards[idx]).to(self.device)

        # Convert images to NCHW (PyTorch format)
        images = images.permute(0, 3, 1, 2)
        next_images = next_images.permute(0, 3, 1, 2)

        return (
            (images, skills, robot_states),
            actions,
            rewards,
            (next_images, next_skills, next_robot_states),
        )
