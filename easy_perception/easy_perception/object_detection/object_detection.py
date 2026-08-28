#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from easy_interfaces.msg import BoundingBoxes, BoundingBox
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

import cv2
import numpy as np

import threading
import websockets
import asyncio

from easy_perception.object_detection.yolo_detection import YoloDetection
from easy_perception.webrtc.signaling_server import WebRTCSignalingServer


class ObjectDetectionNode(Node):
    def __init__(self):
        super().__init__("object_detection_node")

        self.bridge = CvBridge()
        
        # Detection model
        self.detector = YoloDetection(self)

        # Signaling server
        self.server = WebRTCSignalingServer(
            ["color", "attention"]
        )
        
        # Publisher
        self.bboxes_pub = self.create_publisher(BoundingBoxes, "easy_object_detection/bounding_boxes", 1)

        # Subscribers
        self.rgb_sub = self.create_subscription(
            Image, "camera/color", self.rgb_callback, 1
        )

        self.attention_sub = self.create_subscription(
            Image, "/heatmap", self.attention_callback, 1
        )

        self.get_logger().info(f"Start detection")

    def rgb_callback(self, msg: Image):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            self.server.update_frame("color", frame)
            
            results = self.detector.detect(frame)
            boxes_msg = BoundingBoxes()
            for det in results:
                box_msg = BoundingBox()
                box_msg.x = int(det['bbox'][0])
                box_msg.y = int(det['bbox'][1])
                box_msg.w = int(det['bbox'][2])
                box_msg.h = int(det['bbox'][3])
                box_msg.confidence = float(det['confidence'])
                box_msg.class_id = det['class']
                boxes_msg.bboxes.append(box_msg)
                
            boxes_msg.header = msg.header
            self.bboxes_pub.publish(boxes_msg)
            
        except Exception as e:
            self.get_logger().error(f"Rgb error: {e}")

    def attention_callback(self, msg: Image):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            # Scale frame to 640x480 for WebRTC
            frame = cv2.resize(frame, (640, 480))
            self.server.update_frame("attention", frame)
        except Exception as e:
            self.get_logger().error(f"Attention error: {e}")

    def depth_to_bgr(self, frame: np.ndarray) -> np.ndarray:
        """
        Convert ROS depth image (32FC1) to 8-bit BGR for WebRTC.
        """
        if frame.dtype == np.float32:
            # Normalize depth to 0-255 for visualization
            frame_norm = np.clip(frame, 0, 1.0)  # max 1.0, adjust as needed
            frame_norm = (frame_norm / frame_norm.max() * 255).astype(np.uint8)
        else:
            frame_norm = frame.astype(np.uint8)

        # Convert to 3-channel BGR
        return cv2.cvtColor(frame_norm, cv2.COLOR_GRAY2BGR)
            

async def main_func():
    rclpy.init()
    node = ObjectDetectionNode()

    executor = rclpy.executors.SingleThreadedExecutor()
    executor.add_node(node)
    
    # Run ROS2 in a separate thread
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    # Start WebSocket server
    async with websockets.serve(node.server.handler, "0.0.0.0", 8765):
        node.get_logger().info("WebRTC signaling server started on ws://0.0.0.0:8765")
        # Keep the server running until cancelled
        await asyncio.Future()  # never completes, keeps event loop alive

    # Cleanup (won’t actually reach here unless cancelled)
    node.destroy_node()
    rclpy.shutdown()


def main():
    asyncio.run(main_func())

