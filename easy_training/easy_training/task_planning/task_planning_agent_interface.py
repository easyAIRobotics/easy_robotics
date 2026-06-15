import os
import threading
import time

from easy_training.agent_interfaces import AgentInterface, ActionInterface, StateInterface
from easy_training.task_planning.task_planning_action_interface import TaskPlanningActionInterface
from easy_training.task_planning.task_planning_state_interface import TaskPlanningStateInterface
from easy_training.task_planning.task_planning_memory import TaskPlanningReplayBuffer
from easy_training.task_planning.task_planning_cfg import *
from easy_training.sac.task_planning_sac_agent import TaskPlanningSACAgent

from rclpy.node import Node
from easy_interfaces.srv import SetString
from std_msgs.msg import String

RL_BUFFER_CAPACITY = 10000
BC_BUFFER_CAPACITY = 10000
VAL_BUFFER_CAPACITY = 2000

class TaskPlanningAgentInterface(AgentInterface):
    def __init__(self, node: Node, agent_name: str = "task_planning"):
        super().__init__(
            node,
            TaskPlanningActionInterface(node),
            TaskPlanningStateInterface(node)
        )
        
        self.buffer_id = 0
        self.buffer_id = 0
        
        self.set_action_service = self._node.create_service(
            srv_type=SetString,
            srv_name=f"{agent_name}/set_action",
            callback=self.set_action_callback
        )
        self.action_publisher = self._node.create_publisher(
            String,
            f"{agent_name}/action",
            1,
        )
        self.action = "pick"
        self.action_interface.set_action(self.action)
        
        self.bc_replay_buffer = TaskPlanningReplayBuffer(BC_BUFFER_CAPACITY)
        self.past_bc_replay_buffer = TaskPlanningReplayBuffer(BC_BUFFER_CAPACITY)
        self.validate_buffer = TaskPlanningReplayBuffer(VAL_BUFFER_CAPACITY)
        
        self.sac_agent = TaskPlanningSACAgent(node)
        
        self.last_load_stamp = time.time()
        self.last_load_folder = "data_0"
        self.load_random_buffers()
        
        if os.path.exists(self.validate_folder):
            self._node.get_logger().info(f"[TaskPlanningAgentInterface] Loading validation buffer from {self.validate_folder}...")
            self.validate_buffer.load_from_disk(self.validate_folder + "/bc_replay_buffer.npz")
            self._node.get_logger().info(f"[TaskPlanningAgentInterface] Loaded {self.validate_buffer.size()} validation samples from disk.")
        else:
            self._node.get_logger().info(f"[TaskPlanningAgentInterface] No existing validation buffer found at {self.validate_folder}, starting with empty validation buffer.")
            
        if os.path.exists(self.model_folder):
            self._node.get_logger().info(f"[TaskPlanningAgentInterface] Loading SAC agent model from {self.model_folder}...")
            self.sac_agent.load_model(self.model_folder)
        else:
            self._node.get_logger().info(f"[TaskPlanningAgentInterface] No existing model found at {self.model_folder}, starting with new agent.")
            
    
    def load_random_buffers(self):
        if os.path.exists(self.buffer_folder):
            # Load folder names in buffer folder
            folder_names = [f for f in os.listdir(self.buffer_folder) if "data_" in f]
            
            # Randomly select one folder
            if len(folder_names) > 0: 
                self.buffer_id = (self.buffer_id + 1) % len(folder_names)
                selected_folder = folder_names[self.buffer_id]
                
                if selected_folder == self.last_load_folder:
                    self._node.get_logger().info(f"[TaskPlanningAgentInterface] Randomly selected the same buffer folder {selected_folder} as last time, skipping reload.")
                    return
                
                _buffer_folder = os.path.join(self.buffer_folder, selected_folder)
                self.last_load_folder = selected_folder

                
                self._node.get_logger().info(f"[TaskPlanningAgentInterface] Loading replay buffers from {self.buffer_folder}...")
                self.past_bc_replay_buffer.load_from_disk(_buffer_folder + "/bc_replay_buffer.npz")
                self._node.get_logger().info(f"[TaskPlanningAgentInterface] Loaded {self.past_bc_replay_buffer.size()} BC samples from disk.")
        else:
            self._node.get_logger().info(f"[TaskPlanningAgentInterface] No existing replay buffer found at {self.buffer_folder}, starting with empty buffers.")
            
    
    def set_action_callback(self, request, response):
        self._node.get_logger().info(f"[TaskPlanningAgentInterface] Received request to set action to: {request.data}")
        self.action = request.data
        self.action_interface.set_action(self.action)
        self.action_publisher.publish(String(data=self.action))
        response.success = True
        response.message = f"Action set to {self.action}"
        return response
    
    
    def infer_action(self, deterministic=True):
        state = self.state_interface.get_state()
        if state is None:
            self._node.get_logger().warning("[TaskPlanningAgentInterface] No state available for action inference.")
            return None
        
        action = self.sac_agent.infer_action(state, deterministic)
        
        if action is not None:
            action = action.tolist()[0]
        else:
            action = []

        return action
    
    
    def update(self):
        # Return if an update is already in progress
        if self._update_thread is not None and self._update_thread.is_alive():
            return
        
        with self._update_lock:
            self._update_thread = threading.Thread(
                target=self._update_worker,
                daemon=True
            )
            self._update_thread.start()

    def _update_worker(self):
        try:
            with self._buffer_lock:
                now = time.time()
                if now - self.last_load_stamp > 60:  # Reload buffers every 1 minute
                    self.load_random_buffers()
                    self.last_load_stamp = now
                
                bc_batch = self.bc_replay_buffer.sample(512, recent=False)
                past_bc_batch = None
                if self.past_bc_replay_buffer is not None:
                    past_bc_batch = self.past_bc_replay_buffer.sample(512, recent=False)
                val_batch = self.validate_buffer.sample(128)

            losses = self.sac_agent.update(bc_batch, past_bc_batch, val_batch)
            self.loss_visualizer.update(losses)

        except Exception as e:
            self._node.get_logger().error(f"[TaskPlanningAgentInterface] Update failed: {e}")
            
    
    def reset(self):
        self._node.get_logger().info(f"[TaskPlanningAgentInterface] Resetting agent state")
        
    
    def add_rl_transition(self, transition):
        # with self._buffer_lock:
        #     self.rl_replay_buffer.add(
        #         transition["rgb_image"],
        #         transition["bbox_list"],
        #         transition["class_list"],
        #         transition["robot_state"],
        #         transition["action"],
        #         transition["reward"],
        #         transition["done"]
        #     )
        return
            
    def add_bc_transition(self, transition):
        with self._buffer_lock:
            self.bc_replay_buffer.add(
                transition["rgb_image"],
                transition["bbox_list"],
                transition["class_list"],
                transition["robot_state"],
                transition["action"],
                transition["reward"],
                transition["done"]
            )
