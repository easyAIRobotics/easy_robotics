import cv2
import rclpy
from rclpy.node import Node

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

from cv_bridge import CvBridge


BASE_FRAME = 'base_link'
TOOL_FRAME = 'virtual_suction_tip'

DOWNSAMPLE_RATIO = 4
DEPTH_SCALE = 1

SKILL_VOCAB = {
    "pick": np.array([1.0, 0.0, 0.0], dtype=np.float32),
    "place": np.array([0.0, 1.0, 0.0], dtype=np.float32),
    "move": np.array([0.0, 0.0, 1.0], dtype=np.float32)
}
        
class SkillExecutionStateInterface(StateInterface):
    def __init__(self, node: Node):
        super().__init__(node)
        
        self.state = {}
        self.depth_transform_mtx = None
        
        # Points publisher (in debug mode)
        self.point_image_pub = self._node.create_publisher(
            PointCloud2,
            "skill_execution_state_interface/points",
            1
        )
        
        # Joint state subscription
        self.state["joint_positions"] = [0.0] * 6
        def _joint_state_callback(msg: JointState):
            self.state["joint_positions"] = msg.position[:6]  # 6-DOF robot arm
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
        self.depth_image = self._node.create_subscription(
            Image,
            "camera/depth",
            self._depth_image_callback,
            1,
            callback_group=self.state_interface_callback_group
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
            self.selected_bbox = msg
        self.bounding_box_sub = self._node.create_subscription(
            BoundingBox,
            "selected_box",
            _bounding_box_callback,
            1,
            callback_group=self.state_interface_callback_group
        )
        
    
    def check_sanity(self) -> bool:
        return not (self.camera_info is None or 
                    self.depth_transform_mtx is None or 
                    self.selected_bbox is None)


    def tf_timer_callback(self):
        try:
            transform = self.tf_buffer.lookup_transform(
                BASE_FRAME,
                TOOL_FRAME,
                rclpy.time.Time(),   # latest available
            )

            self.state["eef_pose"] = [
                transform.transform.translation.x,
                transform.transform.translation.y,
                transform.transform.translation.z,
                transform.transform.rotation.x,
                transform.transform.rotation.y,
                transform.transform.rotation.z,
                transform.transform.rotation.w
            ]
        except (LookupException, ConnectivityException, ExtrapolationException):
            return
    
    def _depth_image_callback(self, msg: Image):
        """Convert depth image to point cloud image and store in state."""
        if self.camera_info is None:
            return
        
        self._lookup_depth_transform()
        if self.depth_transform_mtx is None:
            return
        
        if self.selected_bbox is None:
            return
        
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
        
        bb_min_x = int((self.selected_bbox.x - self.selected_bbox.w / 2) / DOWNSAMPLE_RATIO)
        bb_min_y = int((self.selected_bbox.y - self.selected_bbox.h / 2) / DOWNSAMPLE_RATIO)
        bb_max_x = int((self.selected_bbox.x + self.selected_bbox.w / 2) / DOWNSAMPLE_RATIO)
        bb_max_y = int((self.selected_bbox.y + self.selected_bbox.h / 2) / DOWNSAMPLE_RATIO)
        
        mask = np.zeros(points_image.shape[:2], dtype=np.float32)
        mask[bb_min_y:bb_max_y, bb_min_x:bb_max_x] = 1.0
        
        self.state['point_image'] = np.concatenate(
            [points_image, mask[..., np.newaxis]],
            axis=-1
        )
        
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
        """Publish the point cloud image for visualization/debugging."""
        if self.state['point_image'] is None:
            return
        
        # Convert point image to PointCloud2
        # Flatten to (N, 4) where 4 = (x, y, z, mask)
        points = self.state['point_image'].reshape(-1, 4)

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
        
        
    # Get current states and observations
    def get_state(self) -> dict:
        return self.state
    
    
    def get_reward(self) -> float:
        reward = 0.0
        
        # Reward for closer distance to target (if target is defined in state)
        return reward
    
    
    def get_image(self):
        return self.state["point_image"]
    
    def get_skill(self):
        return SKILL_VOCAB[self.action]
    
    def get_robot_state(self):
        return np.array(self.state["eef_pose"] + [self.state["suction_state"]], dtype=np.float32)
        