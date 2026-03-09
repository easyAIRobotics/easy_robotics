import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup

from std_msgs.msg import Bool
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import TransformStamped
from easy_interfaces.msg import Pixel
from std_srvs.srv import SetBool
from easy_interfaces.srv import ExecuteGoal

import numpy as np
import time
import tf_transformations
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException
from tf2_ros import TransformBroadcaster

BASE_FRAME = "base_link"
WINDOW_SIZE = 15   # must be odd

class PickingExpert:
    def __init__(self, node, tf2_buffer):
        self._node = node
        self._node.get_logger().info("PickingExpert initialized")
        self._tf2_buffer = tf2_buffer
        
        self.depth_image = None
        self.camera_info = None
        self._enable = False
        self._running = False
        
        self._camera_transform = None
        self._tf_broadcaster = TransformBroadcaster(self._node)
        self.picking_callback_group = ReentrantCallbackGroup()
        self.client_callback_group = ReentrantCallbackGroup()
        
        self.suction_cmd_pub = self._node.create_publisher(
            Bool,
            "cmd_suction",
            1,
            callback_group=self.picking_callback_group
        )
        
        def _camera_info_callback(msg: CameraInfo):
            self.camera_info = msg
        self.camera_info_sub = self._node.create_subscription(
            CameraInfo,
            "/camera/info",
            _camera_info_callback,
            1,
            callback_group=self.picking_callback_group
        )
        
        self.selected_point_sub = self._node.create_subscription(
            Pixel,
            "/selected_point",
            self._selected_point_callback,
            1,
            callback_group=self.picking_callback_group
        )
        
        self.enable_picking_srv = self._node.create_service(
            SetBool,
            "picking_expert/enable",
            self._enable_picking_callback,
            callback_group=self.picking_callback_group
        )
        
        self.execute_goal_client = self._node.create_client(
            ExecuteGoal,
            "execute_goal",
            callback_group=self.client_callback_group
        )
        
        self._node.get_logger().info("PickingExpert setup complete")
        
    def update_depth_image(self, depth_image):
        self.depth_image = depth_image
        

    def _enable_picking_callback(self, request, response):
        self._enable = request.data
        response.success = True
        response.message = f"Picking expert {'enabled' if self._enable else 'disabled'}"
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
            pre_picking_point,picking_point, picking_quat = self._compute_picking_pose(point)

            if picking_point is not None and picking_quat is not None:

                self._node.get_logger().info(
                    f"Computed picking pose: position={picking_point}, orientation={picking_quat}"
                )
                
                goal_req = ExecuteGoal.Request()
                goal_req.speed_factor = 0.2
                goal_req.goal.header.frame_id = BASE_FRAME
                goal_req.goal.header.stamp = self._node.get_clock().now().to_msg()
                goal_req.goal.pose.position.x = float(pre_picking_point[0])
                goal_req.goal.pose.position.y = float(pre_picking_point[1])
                goal_req.goal.pose.position.z = float(pre_picking_point[2])
                goal_req.goal.pose.orientation.x = float(picking_quat[0])
                goal_req.goal.pose.orientation.y = float(picking_quat[1])
                goal_req.goal.pose.orientation.z = float(picking_quat[2])
                goal_req.goal.pose.orientation.w = float(picking_quat[3])
                    
                future = self.execute_goal_client.call_async(goal_req)
                while not future.done():
                    time.sleep(0.001)
                    
                if future.result() is None or not future.result().success:
                    self._running = False
                    return
                
                goal_req.goal.pose.position.x = float(picking_point[0])
                goal_req.goal.pose.position.y = float(picking_point[1])
                goal_req.goal.pose.position.z = float(picking_point[2])
                
                future = self.execute_goal_client.call_async(goal_req)
                while not future.done():
                    time.sleep(0.001)
                    
                if future.result() is None or not future.result().success:
                    self._running = False
                    return

                self.suction_cmd_pub.publish(Bool(data=True))

            else:
                self._node.get_logger().warn("Failed to compute picking pose")
        
        self._running = False
        
            
    def _compute_picking_pose(self, pixel):
        """Compute the 3D picking point and orientation from the selected pixel."""
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
        # 2) Compute picking point
        # ---------------------------
        picking_point_camera = points_camera[len(points_camera) // 2]  # center point in camera frame

        point_camera_h = np.append(picking_point_camera, 1.0)
        point_base = self._camera_transform @ point_camera_h
        picking_point = point_base[:3]

        # ---------------------------
        # 3) Compute surface normal using SVD
        # ---------------------------
        picking_point_camera, normal_camera = self.fit_plane_lstsq_with_outlier_rejection(points_camera)

        # ---------------------------
        # 4) Transform normal to base frame
        # ---------------------------
        R_cam = self._camera_transform[:3, :3]
        normal_base = R_cam @ normal_camera
        normal_base /= np.linalg.norm(normal_base)

        # ---------------------------
        # 5) Build rotation matrix
        #    Z axis = normal
        # ---------------------------
        z_new = normal_base

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
        picking_quat = tf_transformations.quaternion_from_matrix(T)
        picking_point += 0.01 * normal_base  # offset along normal for better grasping
        pre_picking_point = picking_point - 0.01 * normal_base  # pre-picking point for approach

        return pre_picking_point, picking_point, picking_quat
    

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
            
            
    def fit_plane_lstsq_with_outlier_rejection(self, points, threshold=0.005, max_iters=3):
        """
        Fit a plane to Nx3 points using least squares, with iterative outlier rejection.
        Args:
            points: Nx3 numpy array
            threshold: max distance from plane to keep point (meters)
            max_iters: number of iterations
        Returns:
            picking_point_camera: 3D center of patch (median)
            normal_camera: unit normal vector
        """
        if points.shape[0] < 3:
            return None, None

        points_working = points.copy()

        for _ in range(max_iters):
            # Least squares plane fit: z = a*x + b*y + c
            X = points_working[:, 0]
            Y = points_working[:, 1]
            Z = points_working[:, 2]

            A = np.c_[X, Y, np.ones_like(X)]
            plane_coeffs, _, _, _ = np.linalg.lstsq(A, Z, rcond=None)
            a, b, c = plane_coeffs

            # Compute residuals (distance along z)
            Z_pred = a * X + b * Y + c
            residuals = np.abs(Z - Z_pred)

            # Keep inliers
            inliers = residuals < threshold
            points_working = points_working[inliers]

            # Stop if too few points
            if points_working.shape[0] < 3:
                return None, None

        # Compute normal vector from plane coefficients
        normal_camera = np.array([a, b, -1.0])
        normal_camera /= np.linalg.norm(normal_camera)

        # Optional: enforce direction toward camera Z
        if np.dot(normal_camera, np.array([0.0, 0.0, 1.0])) < 0:
            normal_camera = -normal_camera

        # Picking point = median of inliers
        picking_point_camera = np.median(points_working, axis=0)

        return picking_point_camera, normal_camera
