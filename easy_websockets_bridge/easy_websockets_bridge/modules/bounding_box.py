import asyncio
from easy_interfaces.msg import BoundingBoxes, BoundingBox, Pixel
from rclpy.node import Node

class BoundingBoxModule:
    def __init__(self, node: Node, ws_server):
        self._node = node
        self.ws = ws_server

        self.sub = self._node.create_subscription(
            BoundingBoxes,
            "/detections",
            self._on_detections,
            10,
        )
        
        self.selected_box_pub = self._node.create_publisher(
            BoundingBox,
            "/selected_box",
            1,
        )
        
        self.selected_box_sub = self._node.create_subscription(
            BoundingBox,
            "/selected_box",
            self._on_selected_box,
            10,
        )
        
        self.selected_point_pub = self._node.create_publisher(
            Pixel,
            "/selected_point",
            1,
        )

        self._node.get_logger().info("BoundingBoxModule initialized")

    def _on_detections(self, msg: BoundingBoxes):
        boxes = []
        count = 0

        for box in msg.bboxes:
            boxes.append({
                "id": count,
                "label": box.class_id,
                "score": float(box.confidence),
                "x": int(box.x),
                "y": int(box.y),
                "w": int(box.w),
                "h": int(box.h),
            })
            count += 1

        payload = {
            "type": "bounding_boxes",
            "camera": msg.header.frame_id,
            "stamp": (
                msg.header.stamp.sec
                + msg.header.stamp.nanosec * 1e-9
            ),
            "boxes": boxes,
        }
        
        self._node.send_ws(payload)
        
    def _on_selected_box(self, msg: BoundingBox):
        payload = {
            "type": "selected_box",
            "box": {
                "label": msg.class_id,
                "score": float(msg.confidence),
                "x": int(msg.x),
                "y": int(msg.y),
                "w": int(msg.w),
                "h": int(msg.h),
            },
        }
        self._node.send_ws(payload)

    # Called later for inbound WS messages
    def handle_ws_message(self, data: dict):
        if data.get("type") == "selected_box":        
            box = data["box"]
            self._node.get_logger().info(
                f"Selected box received: {box}"
            )

            msg = BoundingBox()
            msg.class_id = box["label"]
            msg.confidence = box["confidence"]
            msg.x = box["x"]
            msg.y = box["y"]
            msg.w = box["w"]
            msg.h = box["h"]

            self.selected_box_pub.publish(msg)
        if data.get("type") == "selected_point":
            point = data["point"]
            self._node.get_logger().info(
                f"Selected point received: {point}"
            )
            
            msg = Pixel()
            msg.px = point["x"]
            msg.py = point["y"]

            self.selected_point_pub.publish(msg)