import torch
import numpy as np

from easy_training.agent_interfaces import ReplayBuffer

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
        self.dones = np.zeros((capacity, 1), dtype=np.float32)

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
        done
    ):
        self.images[self.ptr] = image
        self.skills[self.ptr] = skill
        self.robot_states[self.ptr] = robot_state
        self.dones[self.ptr] = done
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward

        self.next_images[self.ptr] = next_image
        self.next_skills[self.ptr] = next_skill
        self.next_robot_states[self.ptr] = next_robot_state

        self.ptr = (self.ptr + 1) % self.capacity
        self.buffer_size = min(self.buffer_size + 1, self.capacity)

    # ======================================================
    # Sample batch
    # ======================================================
    def sample(self, batch_size):
        if self.buffer_size == 0:
            return None
            
        idx = np.random.randint(0, self.buffer_size, size=batch_size)

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
        dones = torch.from_numpy(self.dones[idx]).to(self.device)

        # Convert images to NCHW (PyTorch format)
        images = images.permute(0, 3, 1, 2)
        next_images = next_images.permute(0, 3, 1, 2)

        return (
            (images, skills, robot_states),
            actions,
            rewards,
            dones,
            (next_images, next_skills, next_robot_states),
        )


    def sample_all(self):
        if self.buffer_size == 0:
            return None
        
        # -------- Current --------
        images = torch.from_numpy(self.images[:self.buffer_size]).to(self.device)
        skills = torch.from_numpy(self.skills[:self.buffer_size]).to(self.device)
        robot_states = torch.from_numpy(self.robot_states[:self.buffer_size]).to(self.device)
        # -------- Next --------
        next_images = torch.from_numpy(self.next_images[:self.buffer_size]).to(self.device)
        next_skills = torch.from_numpy(self.next_skills[:self.buffer_size]).to(self.device)
        next_robot_states = torch.from_numpy(self.next_robot_states[:self.buffer_size]).to(self.device)
        
        actions = torch.from_numpy(self.actions[:self.buffer_size]).to(self.device)
        rewards = torch.from_numpy(self.rewards[:self.buffer_size]).to(self.device)
        dones = torch.from_numpy(self.dones[:self.buffer_size]).to(self.device)
        # Convert images to NCHW (PyTorch format)
        images = images.permute(0, 3, 1, 2)
        next_images = next_images.permute(0, 3, 1, 2)
        
        return (
            (images, skills, robot_states),
            actions,
            rewards,
            dones,
            (next_images, next_skills, next_robot_states),
        )
        

    def size(self):
        return self.buffer_size


    def save_to_disk(self, file_path: str):
        np.savez_compressed(
            file_path,
            images=self.images[:self.buffer_size],
            skills=self.skills[:self.buffer_size],
            robot_states=self.robot_states[:self.buffer_size],
            actions=self.actions[:self.buffer_size],
            rewards=self.rewards[:self.buffer_size],
            dones=self.dones[:self.buffer_size],
            next_images=self.next_images[:self.buffer_size],
            next_skills=self.next_skills[:self.buffer_size],
            next_robot_states=self.next_robot_states[:self.buffer_size],
        )
        
        
    def load_from_disk(self, file_path: str):
        data = np.load(file_path)
        self.images[:data['images'].shape[0]] = data['images']
        self.skills[:data['skills'].shape[0]] = data['skills']
        self.robot_states[:data['robot_states'].shape[0]] = data['robot_states']
        self.actions[:data['actions'].shape[0]] = data['actions']
        self.rewards[:data['rewards'].shape[0]] = data['rewards']
        self.dones[:data['dones'].shape[0]] = data['dones']
        self.next_images[:data['next_images'].shape[0]] = data['next_images']
        self.next_skills[:data['next_skills'].shape[0]] = data['next_skills']
        self.next_robot_states[:data['next_robot_states'].shape[0]] = data['next_robot_states']
        
        self.buffer_size = data['images'].shape[0]
        self.ptr = self.buffer_size % self.capacity
