from rclpy.node import Node
from cv_bridge import CvBridge
import cv2

from easy_training.agent_interfaces import StateInterface, AgentMode
from easy_training.utils import *
from easy_training.task_planning.task_planning_cfg import *
from easy_training.task_planning.text_embedding import ClassTextEmbedding

from std_msgs.msg import Bool
from sensor_msgs.msg import Image
from easy_interfaces.msg import BoundingBoxes, BoundingBox

class TaskPlanningStateInterface(StateInterface):
    def __init__(self, node: Node):
        super().__init__(node)
        
        self.state = {
            "bbox_list": [[0.0, 0.0, 0.0, 0.0]] * NUM_HEADS,
            "class_list": [[0.0] * TEXT_EMBEDDING_DIM] * NUM_HEADS,
            "robot_state": [0.0],
            "rgb_image": None,
            "done": 0.0,
        }
        self.selected_bbox_center = None
        
        self.class_text_embedding = ClassTextEmbedding()
        
        self.bboxes_subscriber = self._node.create_subscription(
            BoundingBoxes,
            "/easy_object_detection/bounding_boxes",
            self._bboxes_callback,
            1,
            callback_group=self.state_interface_callback_group
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
        self.cv_bridge = CvBridge()
        
        self.reward = 0.0
        self.done = False
        
    def check_sanity(self):
        return self.selected_bbox_center is not None and self.state["rgb_image"] is not None
        
    def _bboxes_callback(self, msg: BoundingBoxes):
        i = 1
        for box in msg.bboxes:
            bbox = [box.x, box.y, box.w, box.h]
            class_name = box.class_id
            self.state["bbox_list"][i] = bbox
            self.state["class_list"][i] = self.class_text_embedding.encode([class_name])[0]
            i += 1
            
        for j in range(i, NUM_HEADS):
            self.state["bbox_list"][j] = [0.0, 0.0, 0.0, 0.0]
            self.state["class_list"][j] = [0.0] * TEXT_EMBEDDING_DIM
            
    def _rgb_image_callback(self, msg: Image):
        rgb_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        # Resize to 224x224
        rgb_image = cv2.resize(rgb_image, (224, 224))
        self.state["rgb_image"] = rgb_image
        
    def _selected_bbox_callback(self, msg: BoundingBox):
        self.selected_bbox_center = [msg.x, msg.y]
        print(f"Selected bbox center updated to: {self.selected_bbox_center}", flush=True)
                
    def set_action(self, action):
        self.action = action
        self.state["done"] = 0.0
        
    def get_rgb_image(self):
        return self.state["rgb_image"]
        
    def get_bbox_list(self):
        return self.state["bbox_list"]
    
    def get_class_list(self):
        return self.state["class_list"]
    
    def get_robot_state(self):
        return self.state["robot_state"]
    
    def get_selected_bbox_center(self):
        return self.selected_bbox_center
    
    def get_reward(self):
        return 0.0
    
    def get_done(self):
        return self.state["done"]
    
    def get_state(self) -> dict:
        return self.state
