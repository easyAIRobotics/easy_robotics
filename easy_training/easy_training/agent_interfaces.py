import time
from enum import Enum
from rclpy.callback_groups import ReentrantCallbackGroup

import rclpy
from rclpy.node import Node

from easy_training.utils import ReplayBuffer
        
    
class StateInterface:
    def __init__(self, node: Node):
        self._node = node
        self.state_interface_callback_group = ReentrantCallbackGroup()

    def check_sanity(self) -> bool:
        raise NotImplementedError("[StateInterface] The check_sanity method must be implemented by the subclass.")
    
        # Get current states and observations
    def get_state(self) -> dict:
        raise NotImplementedError("[StateInterface] The get_state method must be implemented by the subclass.")
    
    def get_reward(self) -> float:
        raise NotImplementedError("[StateInterface] The get_reward method must be implemented by the subclass.")

class ActionInterface:
    def __init__(self, node: Node):
        self._node = node
        self.frequency = 10.0  # Default frequency for action execution
        self.action_interface_callback_group = ReentrantCallbackGroup()
    
    # Perform action and return the reward"""
    def perform(self, act: dict, state_interface: StateInterface) -> tuple[float, dict]:
        raise NotImplementedError("[ActionInterface] The perform method must be implemented by the subclass.")
    
    def wait_for_next_state(self):
        # Wait for the next state update based on the specified frequency
        time.sleep(1.0 / self.frequency)

class AgentMode(Enum):
    IDLE = 0
    SELF_LEARNING = 1
    BEHAVIOR_CLONING = 2
    PERFORMING = 3
    

class AgentInterface:
    def __init__(self, node: Node,
                 action_interface: ActionInterface, 
                 state_interface: StateInterface):
        self._node = node
        
        self.action_interface = action_interface
        self.state_interface = state_interface
        self.mode_ = AgentMode.IDLE
        
        self.replay_buffer = ReplayBuffer(100)
    
        
    def train_loop(self):
        while rclpy.ok():
            reward = 0.0
            if not self.state_interface.check_sanity():
                print("[AgentInterface] State sanity check failed. Waiting for next state update...", flush=True)
                self.action_interface.wait_for_next_state()
                continue
            
            current_state = self.state_interface.get_state()
            
            if self.mode_ == AgentMode.IDLE:
                print("[AgentInterface] Agent is idle. Waiting for mode change...", flush=True)
                time.sleep(1.0)
                continue
            
            elif self.mode_ == AgentMode.SELF_LEARNING:
                print("[AgentInterface] self-learning mode, make an exploration action", flush=True)
                action = self.infer_action(current_state)
                reward, executed_action = self.action_interface.perform(action, self.state_interface)
                self.replay_buffer.push((current_state, action, reward))
                self.update()
                
            elif self.mode_ == AgentMode.BEHAVIOR_CLONING:
                print("[AgentInterface] behavior cloning mode, perform no action", flush=True)
                reward, executed_action = self.action_interface.perform({}, self.state_interface)
                self.replay_buffer.push((current_state, executed_action, reward))
                self.update()
                
            elif self.mode_ == AgentMode.PERFORMING:
                print("[AgentInterface] performing mode, make a policy-generated action", flush=True)
                action = self.infer_action(current_state)
                reward, executed_action = self.action_interface.perform(action, self.state_interface)
            
            print(f"[AgentInterface] Received reward: {reward}", flush=True)
        self.reset()
        
        
    def set_mode(self, mode: str):
        print(f"[AgentInterface] Setting mode to: {mode}", flush=True)
        if mode == "training/auto":
            self.mode_ = AgentMode.SELF_LEARNING
        elif mode == "training/manual":
            self.mode_ = AgentMode.BEHAVIOR_CLONING
        elif mode == "training/stop":
            self.mode_ = AgentMode.IDLE
        elif mode == "performing":
            self.mode_ = AgentMode.PERFORMING
        else:
            print(f"[AgentInterface] Unknown mode: {mode}. Defaulting to IDLE.", flush=True)
            self.mode_ = AgentMode.IDLE
            
        self.action_interface.set_mode(self.mode_)
        self.state_interface.set_mode(self.mode_)
    
    
    def infer_action(self, state: dict) -> dict:
        raise NotImplementedError("[AgentInterface] The infer_action method must be implemented by the subclass.")
    
    
    def update(self):
        raise NotImplementedError("[AgentInterface] The update method must be implemented by the subclass.")
    
        
    def reset(self):
        raise NotImplementedError("[AgentInterface] The reset method must be implemented by the subclass.")

