import cv2
import rclpy
from rclpy.node import Node

from easy_training.skill_execution.collision_monitor import CollisionMonitor
from easy_training.agent_interfaces import ActionInterface, AgentMode, StateInterface
from easy_training.utils import *

from std_msgs.msg import Bool, Header
from sensor_msgs.msg import JointState, Image, CameraInfo, PointCloud2, PointField
from geometry_msgs.msg import PoseStamped
from easy_interfaces.msg import BoundingBox
import sensor_msgs_py.point_cloud2 as pc2
from tf2_ros import Buffer, TransformListener
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException

import numpy as np
import tf_transformations
from threading import Lock

from cv_bridge import CvBridge


BASE_FRAME = 'base_link'
TOOL_FRAME = 'virtual_suction_tip'

DOWNSAMPLE_RATIO = 4
DEPTH_SCALE = 1
TARGET_IMAGE_SIZE = (16, 16)

SKILL_VOCAB = {
    "pick": np.array([1.0, 0.0, 0.0], dtype=np.float32),
    "place": np.array([0.0, 1.0, 0.0], dtype=np.float32),
    "move": np.array([0.0, 0.0, 1.0], dtype=np.float32)
}

SUCCESS_PICK_REWARD = 10.0
FAILED_PICK_PENALTY = 0.0

SUCCESS_DROP_REWARD = 10.0
FAILED_DROP_PENALTY = -10.0

HEIGHT_REWARD_SCALE = 1.0
DISTANCE_REWARD_SCALE = 2.0
DROPPING_HEIGHT = 0.3
        
class SkillExecutionStateInterface(StateInterface):
    def __init__(self, node: Node):
        super().__init__(node)
        
        self.mutex = Lock()
        self.state = {"done": 0.0, "skill_vector": np.zeros(3, dtype=np.float32)}
        self.depth_transform_mtx = None
        
        # Points publisher (in debug mode)
        self.point_image_pub = self._node.create_publisher(
            PointCloud2,
            "skill_execution_state_interface/points",
            1
        )
        
        self.result_pub = self._node.create_publisher(
            Bool,
            "skill_execution/result",
            1
        )
        
        # Joint state subscription
        self.state["joint_positions"] = [0.0] * 6
        def _joint_state_callback(msg: JointState):
            self.state["joint_positions"] = list(msg.position[:6])  # 6-DOF robot arm
        self.joint_state_sub = self._node.create_subscription(
            JointState,
            "joint_states",
            _joint_state_callback,
            1,
            callback_group=self.state_interface_callback_group
        )
        
        # TF listener for end-effector pose
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self._node)
        self.tf_timer = self._node.create_timer(0.01, self.tf_timer_callback, callback_group=self.state_interface_callback_group)
        
        # Gripper state subscription
        self.state["suction_state"] = 0.0
        def _suction_state_callback(msg: Bool):
            self.state["suction_state"] = 0.0 if not msg.data else 1.0
        self.suction_state_sub = self._node.create_subscription(
            Bool,
            "suction_state",
            _suction_state_callback,
            1,
            callback_group=self.state_interface_callback_group
        )
        
        self.state["cmd_suction_state"] = 0.0
        def _cmd_suction_state_callback(msg: Bool):
            self.state["cmd_suction_state"] = 0.0 if not msg.data else 1.0
        self.cmd_suction_state_sub = self._node.create_subscription(
            Bool,
            "cmd_suction_state",
            _cmd_suction_state_callback,
            1,
            callback_group=self.state_interface_callback_group
        )
        
        # Depth observation
        self.state["point_image"] = None
        self.state["target_image"] = None
        self.state["original_target_image"] = None
        self.depth_image = self._node.create_subscription(
            Image,
            "camera/depth",
            self._depth_image_callback,
            1,
            callback_group=self.state_interface_callback_group
        )
        
        self.original_depth_image_pub = self._node.create_publisher(
            Image,
            "skill_execution_state_interface/original_depth",
            1
        )
        
        self.camera_info = None
        def _camera_info_callback(msg: CameraInfo):
            self.camera_info = msg
        self.camera_info_sub = self._node.create_subscription(
            CameraInfo,
            "camera/info",
            _camera_info_callback,
            1,
            callback_group=self.state_interface_callback_group
        )
        self.cv_bridge = CvBridge()
        
        # Bounding box observation
        self.selected_bbox = None
        def _bounding_box_callback(msg: BoundingBox):
            with self.mutex:
                self.selected_bbox = msg
                self.state["original_target_image"] = None
        self.bounding_box_sub = self._node.create_subscription(
            BoundingBox,
            "selected_box",
            _bounding_box_callback,
            1,
            callback_group=self.state_interface_callback_group
        )
        
        # Collision monitor
        self.collision_monitor = CollisionMonitor(node)
        
    
    def check_sanity(self) -> bool:
        return not (self.camera_info is None or 
                    self.depth_transform_mtx is None or 
                    self.selected_bbox is None or
                    self.state["eef_pose"] is None or
                    self.state["joint_positions"] is None or
                    self.state["suction_state"] is None or
                    self.state["cmd_suction_state"] is None or
                    self.state["point_image"] is None or
                    self.state["target_image"] is None or
                    self.state["original_target_image"] is None)


    def tf_timer_callback(self):
        try:
            transform = self.tf_buffer.lookup_transform(
                BASE_FRAME,
                TOOL_FRAME,
                rclpy.time.Time(),   # latest available
            )
            
            rot6d = quaternion_to_6d([
                transform.transform.rotation.x,
                transform.transform.rotation.y,
                transform.transform.rotation.z,
                transform.transform.rotation.w
            ])
            
            self.state["last_eef_pose"] = self.state.get("eef_pose", [0.0] * 9)

            self.state["eef_pose"] = [
                transform.transform.translation.x,
                transform.transform.translation.y,
                transform.transform.translation.z,
            ] + rot6d
            
        except (LookupException, ConnectivityException, ExtrapolationException):
            return
    
    def _depth_image_callback(self, msg: Image):
        """Convert depth image to point cloud image and store in state."""
        with self.mutex:
            if self.camera_info is None:
                return
            
            self._lookup_depth_transform()
            if self.depth_transform_mtx is None:
                return
            
            if self.selected_bbox is None:
                return
            
            if self.state.get('original_target_image', None) is None:
                self.original_depth_image_pub.publish(msg)
            
            # Median filter to reduce noise and image size
            depth_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            depth_image = depth_image.astype(np.float32) * DEPTH_SCALE
            depth_image[~np.isfinite(depth_image)] = 0.0
            depth_ds = median_downsample(depth_image, DOWNSAMPLE_RATIO)
            h, w = depth_ds.shape
            
            # Convert each pixel to 3D point, then store in that pixel
            u = np.arange(w)
            v = np.arange(h)
            uu, vv = np.meshgrid(u, v)

            # Adjust intrinsics for downsampling
            fx = self.camera_info.k[0] / DOWNSAMPLE_RATIO
            fy = self.camera_info.k[4] / DOWNSAMPLE_RATIO
            cx = self.camera_info.k[2] / DOWNSAMPLE_RATIO
            cy = self.camera_info.k[5] / DOWNSAMPLE_RATIO

            Z = depth_ds
            X = (uu - cx) * Z / fx
            Y = (vv - cy) * Z / fy

            d_points_image = np.stack((X, Y, Z), axis=-1)

            h, w, _ = d_points_image.shape

            # Flatten to (N, 3)
            d_points_image = d_points_image.reshape(-1, 3)

            # Convert to homogeneous coordinates (N, 4)
            d_points_image_h = np.concatenate(
                [d_points_image, np.ones((d_points_image.shape[0], 1))],
                axis=1
            )

            # Apply camera -> base transform
            points_base_h = (self.depth_transform_mtx @ d_points_image_h.T).T

            # Reshape back to image layout
            points_base_h = points_base_h[:, :3].reshape(h, w, 3)
            self._append_boundingbox_mask(points_base_h)
            self._publish_point_image()
        
    def _append_boundingbox_mask(self, points_image: np.ndarray) -> np.ndarray:
        """Append a binary mask channel to the point image based on the selected bounding box."""
        if self.selected_bbox is None:
            return
        bb_min_x = int((self.selected_bbox.x - self.selected_bbox.w / 2) / DOWNSAMPLE_RATIO)
        bb_min_y = int((self.selected_bbox.y - self.selected_bbox.h / 2) / DOWNSAMPLE_RATIO)
        bb_max_x = int((self.selected_bbox.x + self.selected_bbox.w / 2) / DOWNSAMPLE_RATIO)
        bb_max_y = int((self.selected_bbox.y + self.selected_bbox.h / 2) / DOWNSAMPLE_RATIO)
        
        self.state['point_image'] = points_image[::4, ::4]
        target_full_size = points_image[bb_min_y:bb_max_y, bb_min_x:bb_max_x]
        self.state['target_image'] = cv2.resize(target_full_size, TARGET_IMAGE_SIZE, interpolation=cv2.INTER_AREA)
        if self.state['original_target_image'] is None:
            self.state['original_target_image'] = self.state['target_image'].copy()
        
    def _lookup_depth_transform(self):
        """Lookup the transform from camera frame to base frame."""
        if self.depth_transform_mtx is not None:
            return
        
        try:
            depth_transform = self.tf_buffer.lookup_transform(
                BASE_FRAME,
                self.camera_info.header.frame_id,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=1)
            )
            # Translation
            t = depth_transform.transform.translation
            translation = np.array([t.x, t.y, t.z])

            # Rotation (quaternion)
            r = depth_transform.transform.rotation
            quaternion = [r.x, r.y, r.z, r.w]

            # Build homogeneous transform
            T = tf_transformations.quaternion_matrix(quaternion)  # 4x4
            T[0:3, 3] = translation

            self.depth_transform_mtx = T
            self._node.get_logger().info(f"Depth transform found: {self.depth_transform_mtx}")
        except (LookupException, ConnectivityException, ExtrapolationException):
            self._node.get_logger().warn("Depth transform not found yet")
            self.depth_transform_mtx = None
        
    def _publish_point_image(self):
        """Publish combined point cloud with mask labels."""
        if self.state['point_image'] is None:
            return

        import numpy as np

        points_image = self.state['point_image']   # (H, W, 3) or (N, 3)
        target_image = self.state.get('target_image', None)  # (h, w, 3) or (M, 3)
        original_target_image = self.state.get('original_target_image', None)  # (h, w, 3) or (M, 3)

        # --- Flatten to (N, 3) ---
        points_bg = points_image.reshape(-1, 3)

        # Mask = 0.0 for background
        mask_bg = np.zeros((points_bg.shape[0], 1), dtype=np.float32)
        points_bg = np.hstack((points_bg, mask_bg))  # (N, 4)

        # --- Target points ---
        if target_image is not None:
            points_target = target_image.reshape(-1, 3)

            # Mask = 1.0 for target
            mask_target = np.ones((points_target.shape[0], 1), dtype=np.float32)
            points_target = np.hstack((points_target, mask_target))  # (M, 4)

            # --- Concatenate ---
            points = np.vstack((points_bg, points_target))
        else:
            points = points_bg
            
        if original_target_image is not None:
            points_original_target = original_target_image.reshape(-1, 3)

            # Mask = 2.0 for original target
            mask_original_target = np.ones((points_original_target.shape[0], 1), dtype=np.float32) * 2.0
            points_original_target = np.hstack((points_original_target, mask_original_target))  # (M, 4)

            # --- Concatenate ---
            points = np.vstack((points, points_original_target))

        # Ensure float32
        points = points.astype(np.float32)

        # --- ROS2 PointCloud2 ---
        header = Header()
        header.stamp = self._node.get_clock().now().to_msg()
        header.frame_id = BASE_FRAME

        fields = [
            PointField(name='x', offset=0,  datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4,  datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8,  datatype=PointField.FLOAT32, count=1),
            PointField(name='mask', offset=12, datatype=PointField.FLOAT32, count=1),
        ]

        pc_msg = pc2.create_cloud(header, fields, points)
        self.point_image_pub.publish(pc_msg)
    
    def set_skill_vector(self, skill_vector):
        self.state["skill_vector"] = skill_vector
        
        # Find closet skill in vocab
        min_dist = float('inf')
        closest_skill = None
        for skill_name, skill_vec in SKILL_VOCAB.items():
            dist = np.linalg.norm(skill_vector - skill_vec)
            if dist < min_dist:
                min_dist = dist
                closest_skill = skill_name
        self.action = closest_skill
    
    def set_action(self, action):
        self.action = action
        self.state["skill_vector"] = SKILL_VOCAB.get(action, np.zeros(3, dtype=np.float32))
        self.state["done"] = 0.0
        
    # Get current states and observations
    def get_state(self) -> dict:
        return self.state
    
    
    def get_reward(self) -> float:
        if self.action is None:
            print("[SkillExecutionStateInterface] No action set, returning reward 0.0", flush=True)
            return 0.0
        
        if self.action == "pick":
            print("[SkillExecutionStateInterface] Calculating picking reward", flush=True)
            return self._get_picking_reward()
        elif self.action == "place":
            print("[SkillExecutionStateInterface] Calculating placing reward", flush=True)
            return self._get_placing_reward()
        
        return 0.0  # Default reward for other actions
    
    
    def get_image(self):
        return self.state["point_image"]
    
    def get_target_image(self):
        return self.state["target_image"]
    
    def get_original_target_image(self):
        return self.state["original_target_image"]
    
    def get_skill(self):
        return self.state["skill_vector"]
    
    def get_robot_state(self):
        # joint_positions_cos = np.cos(self.state["joint_positions"])
        return np.array(
            self.state["eef_pose"] + 
            self.state["joint_positions"] +
            # self.state["last_eef_pose"] + 
            # joint_positions_cos.tolist() +
            [self.state["suction_state"]], dtype=np.float32
        )
        
    def get_suction_state(self):
        return self.state["suction_state"]
        
    def get_done(self):
        return self.state["done"]
        
    def check_collision(self) -> bool:
        return self.collision_monitor.check_collisions()
        
    def _get_picking_reward(self) -> float:
        reward = 0.0
        # Reward based on how close the end-effector is to the object (using point cloud mask)
        if self.state["original_target_image"] is not None:
            object_center = np.mean(self.state["original_target_image"], axis=(0, 1))
            ee_position = np.array(self.state["eef_pose"][:3])
            distance = np.linalg.norm(ee_position - object_center)
            reward -= DISTANCE_REWARD_SCALE * (distance ** 2)  # Closer gets higher reward
            reward += HEIGHT_REWARD_SCALE * (min(1.0, self.state["eef_pose"][2]) - 1.0)
            
        # Additional reward for successful suction (if suction state is on and object is close)
        self.state["done"] = 0.0
        if self.state["cmd_suction_state"] == 1.0:  # If suction command is on
            if self.state["suction_state"] == 1.0:  # If suction state is on (object is picked)
                self.state["done"] = 1.0
                reward += SUCCESS_PICK_REWARD  # Bonus for successful pick
                self.result_pub.publish(Bool(data=True))
            else:
                self.state["done"] = 0.0
                reward += FAILED_PICK_PENALTY  # Penalty for failed pick
                self.result_pub.publish(Bool(data=False))
        print(f"Reward from environment (picking): {reward}", flush=True)
                
        return reward
    
    def _get_placing_reward(self) -> float:
        reward = 0.0
        self.state["done"] = 0.0
        # Reward based on how close the end-effector is to the target place position
        if self.state["original_target_image"] is not None:
            object_center = np.mean(self.state["original_target_image"], axis=(0, 1))
            object_center[2] = DROPPING_HEIGHT
            ee_position = np.array(self.state["eef_pose"][:3])
            diff_vector = ee_position - object_center
            print(f"EE position: {ee_position}, Target place position: {object_center}, Diff: {diff_vector}", flush=True)
            distance = np.linalg.norm(diff_vector)
            reward -= DISTANCE_REWARD_SCALE * (distance ** 2)  # Closer gets higher reward
            # Reward for lifting up the object (encourage the agent to lift up after picking)
            reward += HEIGHT_REWARD_SCALE * (min(1.0, self.state["eef_pose"][2]) - 1.0)
            
            # Additional reward for successful release (Only when the gripper close to the place position and suction is off)
            if self.state["cmd_suction_state"] == 0.0:  # If suction command is off
                if distance < 0.3:
                    reward += SUCCESS_DROP_REWARD  # Bonus for successful place
                    self.state["done"] = 1.0
                    self.result_pub.publish(Bool(data=True))
                    return reward
                else:
                    reward += FAILED_DROP_PENALTY  # Penalty for placing in wrong position
                    self.state["done"] = 0.0
            
            self.result_pub.publish(Bool(data=False))
                        
        return reward
    
    def check_above_target(self) -> bool:
        """Check if the end-effector is above the target object (within a certain threshold)."""
        if self.state["original_target_image"] is None or self.state["eef_pose"] is None:
            return False
        
        object_center = np.mean(self.state["original_target_image"], axis=(0, 1))
        ee_position = np.array(self.state["eef_pose"][:3])
        
        # Check if the end-effector is within a certain distance above the object
        distance_xy = np.linalg.norm(ee_position[:2] - object_center[:2])
        height_diff = ee_position[2] - object_center[2]
        # return False
        return distance_xy < 0.04 and height_diff < 0.07
                        