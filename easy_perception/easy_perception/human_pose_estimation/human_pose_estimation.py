#!/usr/bin/env python3
from easy_perception.human_pose_estimation.sam_3d_body_utils import (
    setup_sam_3d_body, setup_visualizer, 
    visualize_2d_results, visualize_3d_mesh, save_mesh_results, 
    display_results_grid, process_image_with_mask
)

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker

import cv2
import numpy as np
from cv_bridge import CvBridge

class HumanPoseEstimation:
    def __init__(self, node: Node):
        self.node = node
        self.bridge = CvBridge()

        """ Parameters """
        self.node.declare_parameter('hf_repo_id', 'facebook/sam-3d-body-vith')
        repo_path = self.node.get_parameter('hf_repo_id').get_parameter_value().string_value
        

        """ Model Initialization """
        self.model = setup_sam_3d_body(
            hf_repo_id=repo_path,
            detector_name="vitdet",
            segmentor_name="",
            fov_name="moge2",
            device="cuda"
        )

        self.visualizer = setup_visualizer()

        self.pose_est_results = None

        """ Publications """
        self.image_publisher = self.node.create_publisher(
            Image,
            'inference_image',
            10)

        self.marker_publisher = self.node.create_publisher(
            Marker,
            'frame',
            10)
        
        """ Subscriptions """
        self.image_subscription = self.node.create_subscription(
            Image,
            'input_image',
            self.image_callback,
            1)

        self.node.get_logger().info("Human Pose Estimation initialized.")
        
    def image_callback(self, msg: Image):
        """ Pipeline:
            - Convert ROS Image to CV Image
            - Inference 3D Human Pose
            - Publish 2d image results
            - Publish 3d marker results
            - Publish list of joint positions
        """
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        self.pose_est_results = self.model.process_one_image(cv_image)

        self.node.get_logger().info(f'Results: {self.pose_est_results}')
        # inference_image = self.visualize_2d_keyframe(cv_image, self.pose_est_results['keypoints_2d'], self.pose_est_results['scores'])
        # inference_image_msg = self.bridge.cv2_to_imgmsg(inference_image, encoding='bgr8')
        # self.image_publisher.publish(inference_image_msg)

        # norm_keypoints_3d = self._normalize_keypoints(self.pose_est_results['keypoints_3d'])
        # frame_3d_markers = self.visualize_3d_keypoints(norm_keypoints_3d, self.pose_est_results['scores'], frame=msg.header.frame_id)
        # self.marker_publisher.publish(frame_3d_markers)
        
    def visualize_2d_keyframe(self, image: np.ndarray, keypoints_2d: np.ndarray, scores: np.ndarray) -> np.ndarray:
        """ Visualize 2D keyframe on image """
        vis_image = image.copy()
        if keypoints_2d is not None:
            for person_id in range(keypoints_2d.shape[0]):
                for kpt_id in range(keypoints_2d.shape[1]):
                    x, y = keypoints_2d[person_id, kpt_id]
                    score = scores[person_id, kpt_id]
                    if score > self.keypoint_score_threshold:
                        cv2.circle(vis_image, (int(x), int(y)), 3, (0, 255, 0), -1)
        return vis_image

    def visualize_3d_keypoints(self, keypoints_3d: np.ndarray, scores: np.ndarray, frame) -> Marker:
        """ Visualize 3D keypoints on rviz, only visualize arms and hands """
        marker = Marker()
        marker.header.frame_id = frame
        marker.header.stamp = rclpy.clock.Clock().now().to_msg()
        marker.type = Marker.LINE_LIST
        marker.action = Marker.ADD
        marker.scale.x = 0.1
        marker.scale.y = 0.1

        if keypoints_3d is not None:
            for person_id in range(keypoints_3d.shape[0]):
                for link in LINKLIST:
                    joint_a_name, joint_b_name, color = link
                    joint_a_id = JOINTLIST[joint_a_name]
                    joint_b_id = JOINTLIST[joint_b_name]

                    score_a = scores[person_id, joint_a_id]
                    score_b = scores[person_id, joint_b_id]
                    if score_a > self.keypoint_score_threshold and score_b > self.keypoint_score_threshold:
                        point_a = keypoints_3d[person_id, joint_a_id]
                        point_b = keypoints_3d[person_id, joint_b_id]
                        self.node.get_logger().info(f"Person {person_id}: {joint_a_name} to {joint_b_name} - Point A: {point_a}, Point B: {point_b}")
                        marker.points.append(
                           Point(x=float(point_a[0]), y=float(point_a[1]), z=float(point_a[2]))
                        )
                        marker.points.append(
                           Point(x=float(point_b[0]), y=float(point_b[1]), z=float(point_b[2]))
                        )
                        marker.colors.append(color)
                        marker.colors.append(color)
        return marker

    def _normalize_keypoints(self, keypoints: np.ndarray) -> np.ndarray:
        """ Rescale keypoints to fit in a predefined space """
        keypoints /= np.array([100.0, 100.0, 25.0])
        
        """ Convert all keypoints to the shoulder center frame """
        ls = JOINTLIST['left_shoulder']
        rs = JOINTLIST['right_shoulder']

        center_shoulder = (keypoints[:, ls] + keypoints[:, rs]) / 2.0
        translated_keypoints = keypoints - center_shoulder[:, None, :]

        """ Flip the key points if needed """
        flipped_keypoints = translated_keypoints.copy()
        z = flipped_keypoints[..., 2]
        flipped_keypoints[..., 2] = np.where(z > 0.0, -z, z)

        return np.array(flipped_keypoints) + np.array(SHOULDER_CENTER)



def main(args=None):
    rclpy.init(args=args)

    node = rclpy.create_node('human_pose_estimation_node')
    human_pose_estimation = HumanPoseEstimation(node)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
