import threading

from easy_training.task_planning.task_planning_cfg import *
from rclpy.node import Node
from easy_training.agent_interfaces import ActionInterface, StateInterface, AgentMode
from easy_training.utils import *

from easy_interfaces.msg import BoundingBoxes, BoundingBox
from easy_interfaces.srv import SetString
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import Image

import cv2


class TaskPlanningActionInterface(ActionInterface):
    def __init__(self, node: Node):
        super().__init__(node)
        self.skill_vector = np.zeros(3, dtype=np.float32)
        self.frequency = 0.5
        
        self.img_width = 640
        self.img_height = 480
        
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
        
        self.bboxes_subscriber = self._node.create_subscription(
            BoundingBoxes,
            "/easy_object_detection/bounding_boxes",
            self._bboxes_callback,
            1
        )
        
        self.selected_bbox_publisher = self._node.create_publisher(
            BoundingBox,
            "selected_box",
            1
        )
        
        self._last_selected_bbox = None
        self._bbox_mutex = threading.Lock()
        self._bboxes = []
        
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
            "rgb_hand_image": state_interface.get_rgb_hand_image(),
            "robot_state": state_interface.get_robot_state(),
        }
        heatmap = state_interface.get_heatmap()
        skill_vector = self.skill_vector
        reward = 0.0
        if act:
            skill_vector = act[0][0]
            print(f"Performing skill vector: {skill_vector}", flush=True)
            heatmap = act[1]
            # Remove batch/channel dimensions
            heatmap = heatmap[0, 0]
            self._publish_heatmap(heatmap, transition["rgb_image"])
            self.send_skill(skill_vector)
            self._last_selected_bbox, is_last_selected = self._select_bbox_from_heatmap(heatmap)
            if self._last_selected_bbox is None:
                return 0.0, None
            
            if not is_last_selected:
                self.selected_bbox_publisher.publish(self._last_selected_bbox)
            self.skill_execution_mode_client.call_async(SetString.Request(data="training/exec"))
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
        
    def _bboxes_callback(self, msg: BoundingBoxes):
        with self._bbox_mutex:
            self._bboxes = [bbox for bbox in msg.bboxes if self._is_bbox_valid(bbox)]
            
    def _select_bbox_from_heatmap(self, heatmap):
        with self._bbox_mutex:
            if self._last_selected_bbox is not None:
                self._bboxes.append(self._last_selected_bbox)
            if not self._bboxes:
                self._node.get_logger().warning("[TaskPlanningActionInterface] No bounding boxes available for selection.")
                return None, False
            scores = []
            for bbox in self._bboxes:
                # Rescale xywh to match the heatmap size (224x224)
                x = int(bbox.x * 224 / self.img_width)
                y = int(bbox.y * 224 / self.img_height)
                w = int(bbox.w * 224 / self.img_width)
                h = int(bbox.h * 224 / self.img_height)
                
                x1 = int(x - w / 2)
                y1 = int(y - h / 2)
                x2 = int(x + w / 2)
                y2 = int(y + h / 2)

                # Ensure the bounding box is within the attention map dimensions
                x1 = max(0, min(x1, heatmap.shape[1] - 1))
                y1 = max(0, min(y1, heatmap.shape[0] - 1))
                x2 = max(0, min(x2, heatmap.shape[1] - 1))
                y2 = max(0, min(y2, heatmap.shape[0] - 1))

                # Calculate the score as the sum of attention values within the bounding box
                sum_score = np.sum(heatmap[y1:y2, x1:x2])
                print(f"[TaskPlanningActionInterface] Bounding box ({x1}, {y1}, {x2}, {y2}) has sum score: {sum_score}", flush=True)
                mean_score = sum_score / ((y2 - y1) * (x2 - x1) + 0.01)
                scores.append(mean_score)
            if self._last_selected_bbox is not None:
                # Double score for the last selected bounding box to encourage consistency
                scores[-1] *= 2.0
                
            # Prob normalize the scores to get probabilities
            scores = np.array(scores)
            
            # Select the bounding box with the highest score
            selected_index = np.argmax(scores)
            
            is_last_selected = (self._last_selected_bbox is not None and
                            selected_index == len(self._bboxes) - 1)
            
            selected_bbox = self._bboxes[selected_index]
            
            return selected_bbox, is_last_selected
        
    def _is_bbox_valid(self, bbox):
        if bbox.class_id != "green_box" and bbox.class_id != "brown_box":
            return True
        
        if bbox.x > 460 or bbox.x < 180 or bbox.y > 370 or bbox.y < 160:
            return False

        center_x = 320
        center_y = 390

        dist = np.sqrt((bbox.x - center_x) ** 2 + (bbox.y - center_y) ** 2)
        return dist < 230