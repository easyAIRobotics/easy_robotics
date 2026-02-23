import cv2
from ultralytics import YOLO

import rclpy
from rclpy.node import Node


class YoloDetection:
    def __init__(self, node: Node):
        """
        Initialize the YOLO detection model.

        Args:
            ros node (Node): ROS2 node for logging and parameters.
        """
        self._node = node
        
        self._node.declare_parameter("model_path", "yolov8n.pt")
        model_path = self._node.get_parameter("model_path").get_parameter_value().string_value
        self.model = YOLO(model_path)

    def detect(self, image):
        """
        Perform object detection on the input image.

        Args:
            image: Input image for detection.

        Returns:
            List of detected objects with their bounding boxes and confidence scores.
        """
        results = self.model(image, conf=0.5, verbose=False)
        detections = []
        for result in results:
            for box in result.boxes:
                detections.append({
                    'bbox': box.xywh.tolist()[0],
                    'confidence': box.conf.item(),
                    'class': result.names[int(box.cls.item())],
                })
                
        # # use cv2 to draw bounding boxes if needed
        # annotated_frame = results[0].plot()
        # cv2.imshow('YOLO Detection', annotated_frame)
        # cv2.waitKey(1)
        
        return detections