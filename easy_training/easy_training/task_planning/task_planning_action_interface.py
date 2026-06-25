from easy_training.task_planning.task_planning_cfg import *
from rclpy.node import Node
from easy_training.agent_interfaces import ActionInterface, StateInterface, AgentMode
from easy_training.utils import *

from easy_interfaces.msg import BoundingBox
from easy_interfaces.srv import SetString
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import Image

import cv2


class TaskPlanningActionInterface(ActionInterface):
    def __init__(self, node: Node):
        super().__init__(node)
        self.skill_vector = np.zeros(3, dtype=np.float32)
        self.frequency = 0.25
        
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
        self.skill_vector = SKILL_VOCAB.get(action, np.zeros(3, dtype=np.float32))
        skill_vec_msg.data = self.skill_vector.tolist()
        self.skill_vector_publisher.publish(skill_vec_msg)
        print(f"Setting action to: {action}, skill vector: {self.skill_vector}", flush=True)
        
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
            self._publish_heatmap(heatmap, transition["rgb_image"])
            
            # TODO: Perfom action here
            self.wait_for_next_state()
            
        else:
            self.skill_execution_mode_client.call_async(SetString.Request(data="training/manual"))
            self.wait_for_next_state()
        
        reward = state_interface.get_reward()
        act_done = state_interface.get_done()    
        transition.update({
            "done": act_done,
            "heatmap": heatmap,
            "skill": skill_vector
        })
        
        # A tricky thing to avoid unreasonable transition
        print(f"Robot state: {transition['robot_state']}, Skill vector: {skill_vector}", flush=True)
        if transition["robot_state"][0] == 0.0:
            print(f"Robot state is 0.0, skill vector: {skill_vector}", flush=True)
            # skill shouldn't be place [0,1,0]
            if np.array_equal(skill_vector, np.array([0, 1, 0], dtype=np.float32)):
                print(f"Invalid transition detected: robot state is 0.0 but skill vector is place [0,1,0]. Ignoring this transition.", flush=True)
                return 0.0, None
        if transition["robot_state"][0] == 1.0:
            print(f"Robot state is 1.0, skill vector: {skill_vector}", flush=True)
            # skill shouldn't be pick [1,0,0]
            if np.array_equal(skill_vector, np.array([1, 0, 0], dtype=np.float32)):
                print(f"Invalid transition detected: robot state is 1.0 but skill vector is pick [1,0,0]. Ignoring this transition.", flush=True)
                return 0.0, None
        
        return reward, transition
    
    def _publish_heatmap(self, heatmap, rgb_image):
        """
        heatmap:   (H, W), values in [0,1]
        rgb_image: (H, W, 3), uint8 RGB image
        """
        
        # Remove batch/channel dimensions
        heatmap = heatmap[0, 0]

        # Convert heatmap to uint8
        heatmap_uint8 = (heatmap * 255).astype(np.uint8)

        # Create colored heatmap
        heatmap_color = cv2.applyColorMap(
            heatmap_uint8,
            cv2.COLORMAP_JET
        )

        # OpenCV uses BGR
        rgb_bgr = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

        # Overlay heatmap on image
        overlay = cv2.addWeighted(
            rgb_bgr,       # original image
            0.6,
            heatmap_color, # heatmap
            0.4,
            0
        )

        # Convert back to RGB if desired
        overlay_rgb = cv2.cvtColor(
            overlay,
            cv2.COLOR_BGR2RGB
        )

        # Publish
        msg = Image()
        msg.header.stamp = self._node.get_clock().now().to_msg()
        msg.header.frame_id = "heatmap_frame"

        msg.height = overlay_rgb.shape[0]
        msg.width = overlay_rgb.shape[1]
        msg.encoding = "rgb8"
        msg.step = overlay_rgb.shape[1] * 3
        msg.data = overlay_rgb.tobytes()

        self.heatmap_publisher.publish(msg)
        
    
    def send_skill(self, skill_vec):
        skill_msg = Float64MultiArray()
        skill_msg.data = skill_vec
        self.skill_vector_publisher.publish(skill_msg)
