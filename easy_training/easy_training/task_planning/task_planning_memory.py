import torch
import numpy as np

from easy_training.agent_interfaces import ReplayBuffer
from easy_training.task_planning.task_planning_cfg import *

class TaskPlanningReplayBuffer(ReplayBuffer):
    def __init__(
        self,
        capacity: int = 10000,
    ):
        super().__init__(capacity)
        self.rgb_image_buffer = np.zeros((capacity, 224, 224, 3), dtype=np.uint8)
        self.rgb_image_hand_buffer = np.zeros((capacity, 64, 64, 3), dtype=np.uint8)
        self.heatmap_buffer = np.zeros((capacity, 224, 224), dtype=np.float32)
        self.robot_state_buffer = np.zeros((capacity, 1), dtype=np.float32)
        self.skill = np.zeros((capacity, 3), dtype=np.float32)
        self.reward_buffer = np.zeros((capacity, 1), dtype=np.float32)
        self.done_buffer = np.zeros((capacity, 1), dtype=np.float32)
        
    
    def add(
        self,
        rgb_image,
        rgb_image_hand,
        heatmap,
        robot_state,
        skill,
        reward,
        done
    ):
        self.rgb_image_buffer[self.ptr] = rgb_image
        self.rgb_image_hand_buffer[self.ptr] = rgb_image_hand
        self.heatmap_buffer[self.ptr] = heatmap
        self.robot_state_buffer[self.ptr] = robot_state
        self.skill[self.ptr] = skill
        self.reward_buffer[self.ptr] = reward
        self.done_buffer[self.ptr] = done
        
        self.ptr = (self.ptr + 1) % self.capacity
        self.buffer_size = min(self.buffer_size + 1, self.capacity)
    
    
    def size(self):
        return self.buffer_size
    
    
    def sample(self, batch_size: int, recent=False):
        if self.buffer_size == 0:
            return None
        size = min(batch_size, self.buffer_size)
        
        if recent:
            idxs = np.arange(max(0, self.buffer_size - size), self.buffer_size)
        else:
            idxs = np.random.choice(self.buffer_size, size, replace=False)
        
        rgb_image = torch.from_numpy(self.rgb_image_buffer[idxs]).to(self.device)
        rgb_image_hand = torch.from_numpy(self.rgb_image_hand_buffer[idxs]).to(self.device)
        heatmap = torch.from_numpy(self.heatmap_buffer[idxs]).to(self.device)
        robot_state = torch.from_numpy(self.robot_state_buffer[idxs]).to(self.device)
        skill = torch.from_numpy(self.skill[idxs]).to(self.device)
        reward = torch.from_numpy(self.reward_buffer[idxs]).to(self.device)
        done = torch.from_numpy(self.done_buffer[idxs]).to(self.device)
        return (
            rgb_image,
            rgb_image_hand,
            heatmap,
            robot_state,
            skill,
            reward,
            done
        )
        
    
    def sample_all(self):
        if self.buffer_size == 0:
            return None
        
        rgb_image = torch.from_numpy(self.rgb_image_buffer[:self.buffer_size]).to(self.device)
        rgb_image_hand = torch.from_numpy(self.rgb_image_hand_buffer[:self.buffer_size]).to(self.device)
        heatmap = torch.from_numpy(self.heatmap_buffer[:self.buffer_size]).to(self.device)
        robot_state = torch.from_numpy(self.robot_state_buffer[:self.buffer_size]).to(self.device)
        skill = torch.from_numpy(self.skill[:self.buffer_size]).to(self.device)
        reward = torch.from_numpy(self.reward_buffer[:self.buffer_size]).to(self.device)
        done = torch.from_numpy(self.done_buffer[:self.buffer_size]).to(self.device)
        return (
            rgb_image,
            rgb_image_hand,
            heatmap,
            robot_state,
            skill,
            reward,
            done
        )
        

    def save_to_disk(self, file_path: str):
        np.savez_compressed(
            file_path,
            rgb_image_buffer=self.rgb_image_buffer[:self.buffer_size],
            rgb_image_hand_buffer=self.rgb_image_hand_buffer[:self.buffer_size],
            heatmap_buffer=self.heatmap_buffer[:self.buffer_size],
            robot_state_buffer=self.robot_state_buffer[:self.buffer_size],
            skill=self.skill[:self.buffer_size],
            reward_buffer=self.reward_buffer[:self.buffer_size],
            done_buffer=self.done_buffer[:self.buffer_size]
        )
        
    def load_from_disk(self, file_path: str):
        data = np.load(file_path)
        self.rgb_image_buffer[:data['rgb_image_buffer'].shape[0]] = data['rgb_image_buffer']
        self.rgb_image_hand_buffer[:data['rgb_image_hand_buffer'].shape[0]] = data['rgb_image_hand_buffer']
        self.heatmap_buffer[:data['heatmap_buffer'].shape[0]] = data['heatmap_buffer']
        self.robot_state_buffer[:data['robot_state_buffer'].shape[0]] = data['robot_state_buffer']
        self.skill[:data['skill'].shape[0]] = data['skill']
        self.reward_buffer[:data['reward_buffer'].shape[0]] = data['reward_buffer']
        self.done_buffer[:data['done_buffer'].shape[0]] = data['done_buffer']

        self.buffer_size = data['heatmap_buffer'].shape[0]
        self.ptr = self.buffer_size % self.capacity
