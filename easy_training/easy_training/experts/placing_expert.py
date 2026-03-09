import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup

from std_msgs.msg import Bool
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import TransformStamped, PoseStamped
from easy_interfaces.msg import Pixel
from std_srvs.srv import SetBool
from easy_interfaces.srv import ExecuteGoal

import numpy as np
import time
import tf_transformations
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException
from tf2_ros import TransformBroadcaster, Buffer, TransformListener

BASE_FRAME = "base_link"
EEF_FRAME = "virtual_suction_tip"
WINDOW_SIZE = 15   # must be odd
DROPPING_HEIGHT = 0.45

class PlacingExpert:
    def __init__(self, node, tf2_buffer):
        self._node = node
        self._node.get_logger().info("PlacingExpert initialized")
        self._tf2_buffer = tf2_buffer
        self._tf_listener = TransformListener(self._tf2_buffer, self._node)
        
        self.depth_image = None
        self.camera_info = None
        self._enable = False
        self._running = False
        
        self._camera_transform = None
        self._tf_broadcaster = TransformBroadcaster(self._node)
        self.placing_callback_group = ReentrantCallbackGroup()
        self.client_callback_group = ReentrantCallbackGroup()
        
        self.suction_cmd_pub = self._node.create_publisher(
            Bool,
            "cmd_suction",
            1,
            callback_group=self.placing_callback_group
        )
        
        def _camera_info_callback(msg: CameraInfo):
            self.camera_info = msg
        self.camera_info_sub = self._node.create_subscription(
            CameraInfo,
            "/camera/info",
            _camera_info_callback,
            1,
            callback_group=self.placing_callback_group
        )
        
        self.selected_point_sub = self._node.create_subscription(
            Pixel,
            "/selected_point",
            self._selected_point_callback,
            1,
            callback_group=self.placing_callback_group
        )
        
        self.enable_placing_srv = self._node.create_service(
            SetBool,
            "placing_expert/enable",
            self._enable_placing_callback,
            callback_group=self.placing_callback_group
        )
        
        self.execute_goal_client = self._node.create_client(
            ExecuteGoal,
            "execute_goal",
            callback_group=self.client_callback_group
        )
        
        self._node.get_logger().info("PlacingExpert setup complete")
        
    def update_depth_image(self, depth_image):
        self.depth_image = depth_image
        

    def _enable_placing_callback(self, request, response):
        self._enable = request.data
        response.success = True
        response.message = f"placing expert {'enabled' if self._enable else 'disabled'}"
        self._node.get_logger().info(response.message)
        return response
    

    def _selected_point_callback(self, msg: Pixel):
        if self._running:
            return
        
        self._running = True
        point = {"x": msg.px, "y": msg.py}
        self._node.get_logger().info(
            f"Selected point received: {point}"
        )
        
        if self._enable and self.depth_image is not None and self.camera_info is not None:
            placing_point, placing_quat = self._compute_placing_pose(point)

            if placing_point is not None and placing_quat is not None:

                self._node.get_logger().info(
                    f"Computed placing pose: position={placing_point}, orientation={placing_quat}"
                )
                
                current_eef_transform = self._lookup_eef_transform()
                current_eef_transform.pose.position.z += 0.05  # lift up a bit to avoid collision during placing
                current_eef_transform.pose.position.x = max(0.3, current_eef_transform.pose.position.x - 0.05)  # move back a bit
                
                goal_req = ExecuteGoal.Request()
                goal_req.speed_factor = 0.2
                goal_req.goal.header.frame_id = BASE_FRAME
                goal_req.goal.header.stamp = self._node.get_clock().now().to_msg()
                goal_req.goal.pose = current_eef_transform.pose
                
                while not self.execute_goal_client.wait_for_service(timeout_sec=1.0):
                    self._node.get_logger().info("Waiting for execute_goal service...")
                    
                future = self.execute_goal_client.call_async(goal_req)
                while not future.done():
                    time.sleep(0.001)
                if future.result() is None or not future.result().success:
                    self._running = False
                    return
                
                goal_req.goal.pose.position.x = float(placing_point[0])
                goal_req.goal.pose.position.y = float(placing_point[1])
                goal_req.goal.pose.position.z = float(placing_point[2])
                goal_req.goal.pose.orientation.x = float(placing_quat[0])
                goal_req.goal.pose.orientation.y = float(placing_quat[1])
                goal_req.goal.pose.orientation.z = float(placing_quat[2])
                goal_req.goal.pose.orientation.w = float(placing_quat[3])
                
                future = self.execute_goal_client.call_async(goal_req)
                while not future.done():
                    time.sleep(0.001)
                if future.result() is None or not future.result().success:
                    self._running = False
                    return

                self.suction_cmd_pub.publish(Bool(data=False))
            else:
                self._node.get_logger().warn("Failed to compute placing pose")
        
        self._running = False
        
    def _lookup_eef_transform(self):
        try:
            eef_transform = self._tf2_buffer.lookup_transform(
                BASE_FRAME,
                EEF_FRAME,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=1)
            )
            
            eef_pose = PoseStamped()
            eef_pose.header = eef_transform.header
            eef_pose.pose.position.x = eef_transform.transform.translation.x
            eef_pose.pose.position.y = eef_transform.transform.translation.y
            eef_pose.pose.position.z = eef_transform.transform.translation.z
            eef_pose.pose.orientation = eef_transform.transform.rotation
            return eef_pose
        except (LookupException, ConnectivityException, ExtrapolationException):
            self._node.get_logger().warn("EEF transform not found")
            return None
            
    def _compute_placing_pose(self, pixel):
        """Compute the 3D placing point and orientation from the selected pixel."""
        self._lookup_depth_transform()

        half = WINDOW_SIZE // 2

        u0, v0 = pixel["x"], pixel["y"]
        points_camera = []

        # ---------------------------
        # 1) Collect 3D points
        # ---------------------------
        for dv in range(-half, half + 1):
            for du in range(-half, half + 1):
                u = u0 + du
                v = v0 + dv

                if v < 0 or u < 0:
                    continue
                if v >= self.depth_image.shape[0] or u >= self.depth_image.shape[1]:
                    continue

                Z = self.depth_image[v, u]
                if Z <= 0:
                    continue

                X = Z * (u - self.camera_info.k[2]) / self.camera_info.k[0]
                Y = Z * (v - self.camera_info.k[5]) / self.camera_info.k[4]

                points_camera.append([X, Y, Z])

        points_camera = np.array(points_camera)

        if len(points_camera) < 3:
            return None, None

        # ---------------------------
        # 2) Compute placing point
        # ---------------------------
        placing_point_camera = points_camera[len(points_camera) // 2]  # center point in camera frame

        point_camera_h = np.append(placing_point_camera, 1.0)
        point_base = self._camera_transform @ point_camera_h
        placing_point = point_base[:3]
        placing_point[2] += DROPPING_HEIGHT
        picking_dir = placing_point - np.array([0.0, 0.0, placing_point[2] + DROPPING_HEIGHT / 2])  # direction pointing downwards
        picking_dir /= np.linalg.norm(picking_dir)
        # ---------------------------
        # 3) Build rotation matrix
        #    Z axis = normal
        # ---------------------------
        z_new = picking_dir

        # Choose temporary axis not parallel to normal
        x_temp = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(z_new, x_temp)) > 0.9:
            x_temp = np.array([0.0, 1.0, 0.0])

        y_new = np.cross(z_new, x_temp)
        y_new /= np.linalg.norm(y_new)

        x_new = np.cross(y_new, z_new)
        x_new /= np.linalg.norm(x_new)

        # Construct homogeneous transform matrix
        T = np.eye(4)
        T[:3, 0] = x_new
        T[:3, 1] = y_new
        T[:3, 2] = z_new

        # ---------------------------
        # 6) Convert to quaternion
        # ---------------------------
        placing_quat = tf_transformations.quaternion_from_matrix(T)

        return placing_point, placing_quat
    

    def _lookup_depth_transform(self):
        """Lookup the transform from camera frame to base frame."""
        if self._camera_transform is not None:
            return
        
        try:
            depth_transform = self._tf2_buffer.lookup_transform(
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

            self._camera_transform = T
            self._node.get_logger().info(f"Depth transform found: {self._camera_transform}")
        except (LookupException, ConnectivityException, ExtrapolationException):
            self._node.get_logger().warn("Depth transform not found yet")
            self._camera_transform = None
