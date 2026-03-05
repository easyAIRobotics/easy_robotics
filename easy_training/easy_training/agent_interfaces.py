import threading
import time
from enum import Enum
from rclpy.callback_groups import ReentrantCallbackGroup

import rclpy
from rclpy.node import Node


class AgentMode(Enum):
    IDLE = 0
    SELF_LEARNING = 1
    BEHAVIOR_CLONING = 2
    PERFORMING = 3
        
    
class StateInterface:
    def __init__(self, node: Node):
        self._node = node
        self._mode = AgentMode.IDLE
        self.action = None
        self.state_interface_callback_group = ReentrantCallbackGroup()

    def check_sanity(self) -> bool:
        raise NotImplementedError("[StateInterface] The check_sanity method must be implemented by the subclass.")
    
    def set_action(self, action: str):
        self.action = action
        
    def set_mode(self, mode: AgentMode):
        self._mode = mode
        self._node.get_logger().info(f"[SkillExecutionStateInterface] Mode set to: {self._mode.name}")
        if self._mode == AgentMode.IDLE:
            self.selected_bbox = None
    
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
        self._mode = AgentMode.IDLE
        self.action = None
        
    def set_action(self, action: str):
        self.action = action
        self._node.get_logger().info(f"[ActionInterface] Action set to: {self.action}")
        
    def set_mode(self, mode: AgentMode):
        self._mode = mode
        self._node.get_logger().info(f"[SkillExecutionStateInterface] Mode set to: {self._mode.name}")
        if self._mode == AgentMode.IDLE:
            self.selected_bbox = None
    
    # Perform action and return the reward"""
    def perform(self, act_vec: list, state_interface: StateInterface) -> tuple[float, dict]:
        raise NotImplementedError("[ActionInterface] The perform method must be implemented by the subclass.")
    
    def wait_for_next_state(self):
        # Wait for the next state update based on the specified frequency
        time.sleep(1.0 / self.frequency)
    

class AgentInterface:
    def __init__(self, node: Node,
                 action_interface: ActionInterface, 
                 state_interface: StateInterface):
        self._node = node
        
        self.action_interface = action_interface
        self.state_interface = state_interface
        self.sac_agent = None
        
        self.mode_ = AgentMode.IDLE
        
        self.rl_replay_buffer = None
        self.bc_replay_buffer = None

        self._update_lock = threading.Lock()
        self._update_thread = None
        self._buffer_lock = threading.Lock()
        
    def train_loop(self):
        while rclpy.ok():
            reward = 0.0
            if not self.state_interface.check_sanity():
                print("[AgentInterface] State sanity check failed. Waiting for next state update...", flush=True)
                self.action_interface.wait_for_next_state()
                continue
                        
            if self.mode_ == AgentMode.IDLE:
                print("[AgentInterface] Agent is idle. Waiting for mode change...", flush=True)
                time.sleep(1.0)
                continue
            
            if self.mode_ == AgentMode.SELF_LEARNING:
                print("[AgentInterface] self-learning mode, make an exploration action", flush=True)
                action = self.infer_action(deterministic=False)
                reward, transition = self.action_interface.perform(action, self.state_interface)
                transition["reward"] = reward  # Use the reward from performing the policy action
                if transition:
                    self.add_rl_transition(transition)
                self.update()
                
            if self.mode_ == AgentMode.BEHAVIOR_CLONING:
                print("[AgentInterface] behavior cloning mode, perform no action", flush=True)
                reward, transition = self.action_interface.perform([], self.state_interface)
                transition["reward"] = reward  # Use the reward from performing the user action
                if transition:
                    self.add_bc_transition(transition)
                self.update()
                
            if self.mode_ == AgentMode.PERFORMING:
                print("[AgentInterface] performing mode, make a policy-generated action", flush=True)
                action = self.infer_action(deterministic=True)
                _, _ = self.action_interface.perform(action, self.state_interface)
            
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
    
    
    def infer_action(self, deterministic=True) -> dict:
        raise NotImplementedError("[AgentInterface] The infer_action method must be implemented by the subclass.")
    
    
    def update(self):
        raise NotImplementedError("[AgentInterface] The update method must be implemented by the subclass.")
    
        
    def reset(self):
        raise NotImplementedError("[AgentInterface] The reset method must be implemented by the subclass.")
    
    
    def add_rl_transition(self, transition: dict):
        raise NotImplementedError("[AgentInterface] The add_rl_transition method must be implemented by the subclass.")
            

    def add_bc_transition(self, transition: dict):
        raise NotImplementedError("[AgentInterface] The add_bc_transition method must be implemented by the subclass.")
