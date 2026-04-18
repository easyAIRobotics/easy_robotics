import torch
import numpy as np

from easy_training.agent_interfaces import ReplayBuffer

J_ACTION_DIM = 7
E_ACTION_DIM = 10
target_image_shape = (16, 16, 3)  # (H, W, C)

class SkillExecutionReplayBuffer(ReplayBuffer):
    def __init__(
        self,
        capacity,
        image_shape,        # (M, N, C)
        skill_dim,
        robot_state_dim,
        action_dim,
        device="cuda"
    ):
        super().__init__(capacity, device)

        M, N, C = image_shape

        # --------------------
        # Current observation
        # --------------------
        self.images = np.zeros((capacity, M, N, C), dtype=np.float32)
        self.target_images = np.zeros((capacity, *target_image_shape), dtype=np.float32)
        self.original_target_images = np.zeros((capacity, *target_image_shape), dtype=np.float32)
        self.skills = np.zeros((capacity, skill_dim), dtype=np.float32)
        self.robot_states = np.zeros((capacity, robot_state_dim), dtype=np.float32)

        # --------------------
        # Next observation
        # --------------------
        self.next_images = np.zeros((capacity, M, N, C), dtype=np.float32)
        self.next_target_images = np.zeros((capacity, *target_image_shape), dtype=np.float32)
        self.next_original_target_images = np.zeros((capacity, *target_image_shape), dtype=np.float32)
        self.next_skills = np.zeros((capacity, skill_dim), dtype=np.float32)
        self.next_robot_states = np.zeros((capacity, robot_state_dim), dtype=np.float32)

        # --------------------
        # RL signals
        # --------------------
        self.j_actions = np.zeros((capacity, J_ACTION_DIM), dtype=np.float32)
        self.e_actions = np.zeros((capacity, E_ACTION_DIM), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)

    # ======================================================
    # Add transition (works for RL and BC)
    # ======================================================
    def add(
        self,
        image,
        target_image,
        original_target_image,
        skill,
        robot_state,
        j_action,
        e_action,
        reward,
        next_image,
        next_target_image,
        next_original_target_image,
        next_skill,
        next_robot_state,
        done
    ):
        self.images[self.ptr] = image
        self.target_images[self.ptr] = target_image
        self.original_target_images[self.ptr] = original_target_image
        self.skills[self.ptr] = skill
        self.robot_states[self.ptr] = robot_state
        self.dones[self.ptr] = done
        self.j_actions[self.ptr] = j_action
        self.e_actions[self.ptr] = e_action
        self.rewards[self.ptr] = reward

        self.next_images[self.ptr] = next_image
        self.next_target_images[self.ptr] = next_target_image
        self.next_original_target_images[self.ptr] = next_original_target_image
        self.next_skills[self.ptr] = next_skill
        self.next_robot_states[self.ptr] = next_robot_state

        self.ptr = (self.ptr + 1) % self.capacity
        self.buffer_size = min(self.buffer_size + 1, self.capacity)

    # ======================================================
    # Sample batch
    # ======================================================
    def sample(self, batch_size, recent=False):
        if self.buffer_size == 0:
            return None
        
        alpha = 0.0  # tune this (higher = more bias to recent)
        if recent:
            alpha = 0.01
        weights = np.exp(alpha * np.arange(self.buffer_size))
        prob = weights / weights.sum()

        idx = np.random.choice(self.buffer_size, size=batch_size, p=prob)

        # -------- Current --------
        images = torch.from_numpy(self.images[idx]).to(self.device)
        target_images = torch.from_numpy(self.target_images[idx]).to(self.device)
        original_target_images = torch.from_numpy(self.original_target_images[idx]).to(self.device)
        skills = torch.from_numpy(self.skills[idx]).to(self.device)
        robot_states = torch.from_numpy(self.robot_states[idx]).to(self.device)

        # -------- Next --------
        next_images = torch.from_numpy(self.next_images[idx]).to(self.device)
        next_target_images = torch.from_numpy(self.next_target_images[idx]).to(self.device)
        next_original_target_images = torch.from_numpy(self.next_original_target_images[idx]).to(self.device)
        next_skills = torch.from_numpy(self.next_skills[idx]).to(self.device)
        next_robot_states = torch.from_numpy(self.next_robot_states[idx]).to(self.device)

        j_actions = torch.from_numpy(self.j_actions[idx]).to(self.device)
        e_actions = torch.from_numpy(self.e_actions[idx]).to(self.device)
        rewards = torch.from_numpy(self.rewards[idx]).to(self.device)
        dones = torch.from_numpy(self.dones[idx]).to(self.device)

        # Convert images to NCHW (PyTorch format)
        images = images.permute(0, 3, 1, 2)
        target_images = target_images.permute(0, 3, 1, 2)
        original_target_images = original_target_images.permute(0, 3, 1, 2)
        next_images = next_images.permute(0, 3, 1, 2)
        next_target_images = next_target_images.permute(0, 3, 1, 2)
        next_original_target_images = next_original_target_images.permute(0, 3, 1, 2)

        return (
            (images, target_images, original_target_images, skills, robot_states),
            j_actions,
            e_actions,
            rewards,
            dones,
            (next_images, next_target_images, next_original_target_images, next_skills, next_robot_states),
        )


    def sample_all(self):
        if self.buffer_size == 0:
            return None
        
        # -------- Current --------
        images = torch.from_numpy(self.images[:self.buffer_size]).to(self.device)
        target_images = torch.from_numpy(self.target_images[:self.buffer_size]).to(self.device)
        original_target_images = torch.from_numpy(self.original_target_images[:self.buffer_size]).to(self.device)
        skills = torch.from_numpy(self.skills[:self.buffer_size]).to(self.device)
        robot_states = torch.from_numpy(self.robot_states[:self.buffer_size]).to(self.device)
        # -------- Next --------
        next_images = torch.from_numpy(self.next_images[:self.buffer_size]).to(self.device)
        next_target_images = torch.from_numpy(self.next_target_images[:self.buffer_size]).to(self.device)
        next_original_target_images = torch.from_numpy(self.next_original_target_images[:self.buffer_size]).to(self.device)
        next_skills = torch.from_numpy(self.next_skills[:self.buffer_size]).to(self.device)
        next_robot_states = torch.from_numpy(self.next_robot_states[:self.buffer_size]).to(self.device)
        
        j_actions = torch.from_numpy(self.j_actions[:self.buffer_size]).to(self.device)
        e_actions = torch.from_numpy(self.e_actions[:self.buffer_size]).to(self.device)
        rewards = torch.from_numpy(self.rewards[:self.buffer_size]).to(self.device)
        dones = torch.from_numpy(self.dones[:self.buffer_size]).to(self.device)
        # Convert images to NCHW (PyTorch format)
        images = images.permute(0, 3, 1, 2)
        next_images = next_images.permute(0, 3, 1, 2)
        
        return (
            (images, target_images, original_target_images, skills, robot_states),
            j_actions,
            e_actions,
            rewards,
            dones,
            (next_images, next_target_images, next_original_target_images, next_skills, next_robot_states),
        )
        

    def size(self):
        return self.buffer_size


    def save_to_disk(self, file_path: str):
        np.savez_compressed(
            file_path,
            images=self.images[:self.buffer_size],
            target_images=self.target_images[:self.buffer_size],
            original_target_images=self.original_target_images[:self.buffer_size],
            skills=self.skills[:self.buffer_size],
            robot_states=self.robot_states[:self.buffer_size],
            j_actions=self.j_actions[:self.buffer_size],
            e_actions=self.e_actions[:self.buffer_size],
            rewards=self.rewards[:self.buffer_size],
            dones=self.dones[:self.buffer_size],
            next_images=self.next_images[:self.buffer_size],
            next_target_images=self.next_target_images[:self.buffer_size],
            next_original_target_images=self.next_original_target_images[:self.buffer_size],
            next_skills=self.next_skills[:self.buffer_size],
            next_robot_states=self.next_robot_states[:self.buffer_size],
        )
        
        
    def load_from_disk(self, file_path: str):
        data = np.load(file_path)
        self.images[:data['images'].shape[0]] = data['images']
        self.target_images[:data['target_images'].shape[0]] = data['target_images']
        self.original_target_images[:data['original_target_images'].shape[0]] = data['original_target_images']
        self.skills[:data['skills'].shape[0]] = data['skills']
        self.robot_states[:data['robot_states'].shape[0]] = data['robot_states']
        self.j_actions[:data['j_actions'].shape[0]] = data['j_actions']
        self.e_actions[:data['e_actions'].shape[0]] = data['e_actions']
        self.rewards[:data['rewards'].shape[0]] = data['rewards']
        self.dones[:data['dones'].shape[0]] = data['dones']
        self.next_images[:data['next_images'].shape[0]] = data['next_images']
        self.next_target_images[:data['next_target_images'].shape[0]] = data['next_target_images']
        self.next_original_target_images[:data['next_original_target_images'].shape[0]] = data['next_original_target_images']
        self.next_skills[:data['next_skills'].shape[0]] = data['next_skills']
        self.next_robot_states[:data['next_robot_states'].shape[0]] = data['next_robot_states']
        
        self.buffer_size = data['images'].shape[0]
        self.ptr = self.buffer_size % self.capacity
