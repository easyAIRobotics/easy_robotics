from rclpy.node import Node
from cv_bridge import CvBridge
import cv2

from easy_training.agent_interfaces import StateInterface, AgentMode
from easy_training.utils import *
from easy_training.task_planning.task_planning_cfg import *

from std_msgs.msg import Bool
from sensor_msgs.msg import Image
from easy_interfaces.msg import BoundingBoxes, BoundingBox

class TaskPlanningStateInterface(StateInterface):
    def __init__(self, node: Node):
        super().__init__(node)
        
        self.state = {
            "heatmap": None,
            "robot_state": [0.0],
            "rgb_image": None,
            "done": 0.0,
        }
        self.img_width = 640
        self.img_height = 480
        
        self.heatmap_publisher = self._node.create_publisher(
            Image,
            "heatmap/reference",
            1
        )
        
        self.selected_bbox_subscriber = self._node.create_subscription(
            BoundingBox,
            "selected_box",
            self._selected_bbox_callback,
            1,
            callback_group=self.state_interface_callback_group
        )
        
        def _suction_state_callback(msg: Bool):
            self.state["robot_state"][0] = 0.0 if not msg.data else 1.0
        self.suction_state_sub = self._node.create_subscription(
            Bool,
            "suction_state",
            _suction_state_callback,
            1,
            callback_group=self.state_interface_callback_group
        )
        
        self.rgb_subscriber = self._node.create_subscription(
            msg_type=Image,
            topic="camera/rgb",
            callback=self._rgb_image_callback,
            qos_profile=1,
            callback_group=self.state_interface_callback_group
        )
        self.rgb_hand_subscriber = self._node.create_subscription(
            msg_type=Image,
            topic="camera/rgb_hand",
            callback=self._rgb_image_hand_callback,
            qos_profile=1,
            callback_group=self.state_interface_callback_group
        )
        self.cv_bridge = CvBridge()
        
        self.reward = 0.0
        self.done = False
        
    def check_sanity(self):
        return self.state["rgb_image"] is not None
            
    def _rgb_image_callback(self, msg: Image):
        rgb_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        # Resize to 224x224
        rgb_image = cv2.resize(rgb_image, (224, 224))
        self.state["rgb_image"] = rgb_image
        self.img_width = msg.width
        self.img_height = msg.height

    def _rgb_image_hand_callback(self, msg: Image):
        rgb_image_hand = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        # Resize to 224x224
        rgb_image_hand = cv2.resize(rgb_image_hand, (64, 64))
        self.state["rgb_image_hand"] = rgb_image_hand
        
        
    def _selected_bbox_callback(self, msg: BoundingBox):
        # Create a heatmap with the same size as the input image (224x224)
        heatmap = np.zeros((224, 224), dtype=np.float32)
        # Fill the heatmap with 1s in the area of the selected bounding box
        # Resize the bounding box coordinates to match the heatmap size
        x = int(msg.x * 224 / self.img_width)
        y = int(msg.y * 224 / self.img_height)
        w = int(msg.w * 224 / self.img_width)
        h = int(msg.h * 224 / self.img_height)
        heatmap[y-h//2:y+h//2, x-w//2:x+w//2] = 1.0
        self.state["heatmap"] = heatmap
        # Publish the heatmap for visualization
        heatmap_msg = Image()
        
        # Convert heatmap to a format suitable for publishing (e.g., as a grayscale image)
        heatmap_normalized = (heatmap * 255).astype(np.uint8)
        heatmap_msg.data = heatmap_normalized.tobytes()
        heatmap_msg.height, heatmap_msg.width = heatmap_normalized.shape
        heatmap_msg.encoding = "mono8"
        
        heatmap_msg.header.stamp = self._node.get_clock().now().to_msg()
        heatmap_msg.header.frame_id = "heatmap_frame"
        self.heatmap_publisher.publish(heatmap_msg)
                
    def set_action(self, action):
        self.action = action
        self.state["done"] = 0.0
        
    def get_rgb_image(self):
        return self.state["rgb_image"]
    
    def get_rgb_hand_image(self):
        return self.state["rgb_image_hand"]
    
    def get_robot_state(self):
        return self.state["robot_state"]

    def get_heatmap(self):
        return self.state["heatmap"]
    
    def get_reward(self):
        return 0.0
    
    def get_done(self):
        return self.state["done"]
    
    def get_state(self) -> dict:
        return self.state
