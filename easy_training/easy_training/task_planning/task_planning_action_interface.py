from easy_training.task_planning.task_planning_cfg import *
from rclpy.node import Node
from easy_training.agent_interfaces import ActionInterface, StateInterface, AgentMode
from easy_training.utils import *

from easy_interfaces.msg import BoundingBox
from easy_interfaces.srv import SetString
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import Image


class TaskPlanningActionInterface(ActionInterface):
    def __init__(self, node: Node):
        super().__init__(node)
        self.skill_vector = np.zeros(3, dtype=np.float32)
        self.frequency = 0.5
        
        self.skill_vector_publisher = self._node.create_publisher(
            Float64MultiArray,
            "selected_skill_vector",
            1
        )
        
        self.heatmap_publisher = self._node.create_publisher(
            Image,
            "heatmap",
            1
        )
        
        self.skill_execution_mode_client = self._node.create_client(
            SetString,
            "skill_execution/set_mode"
        )
        
        self._last_selected_bbox_center = None
        
    def set_action(self, action):
        super().set_action(action)
        skill_vec_msg = Float64MultiArray()
        print(f"Setting action to: {action}, skill vector: {self.skill_vector}", flush=True)
        self.skill_vector = SKILL_VOCAB.get(action, np.zeros(3, dtype=np.float32))
        skill_vec_msg.data = self.skill_vector.tolist()
        self.skill_vector_publisher.publish(skill_vec_msg)
        
    def perform(self, act: tuple, state_interface: StateInterface) -> dict:
        transition = {
            "rgb_image": state_interface.get_rgb_image(),
            "robot_state": state_interface.get_robot_state(),
        }
        heatmap = state_interface.get_heatmap()
        skill_vector = self.skill_vector
        reward = 0.0
        if act:
            skill_vector = act[0]
            heatmap = act[1]
            
            # TODO: Perfom action here
            self.wait_for_next_state()
            
        else:
            self.skill_execution_mode_client.call_async(SetString.Request(data="training/exec"))
            self.wait_for_next_state()
        
        reward = state_interface.get_reward()
        act_done = state_interface.get_done()    
        transition.update({
            "done": act_done,
            "heatmap": heatmap,
            "skill": skill_vector
        })
        
        return reward, transition
    
    def _publish_heatmap(self, heatmap):
        heatmap_msg = Image()
        
        # Convert heatmap to a format suitable for publishing (e.g., as a grayscale image)
        heatmap_normalized = (heatmap * 255).astype(np.uint8)
        heatmap_msg.data = heatmap_normalized.tobytes()
        heatmap_msg.height, heatmap_msg.width = heatmap_normalized.shape
        heatmap_msg.encoding = "mono8"
        
        heatmap_msg.header.stamp = self._node.get_clock().now().to_msg()
        heatmap_msg.header.frame_id = "heatmap_frame"
        self.heatmap_publisher.publish(heatmap_msg)
        
    
    def send_skill(self, skill_vec):
        skill_msg = Float64MultiArray()
        skill_msg.data = skill_vec
        self.skill_vector_publisher.publish(skill_msg)
