import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from cv_bridge import CvBridge
from tf2_ros import Buffer, TransformListener

from sensor_msgs.msg import Image

from easy_training.experts.picking_expert import PickingExpert
from easy_training.experts.placing_expert import PlacingExpert
    
class ExpertsNode:
    def __init__(self, node):
        self._node = node      
        self.depth_img_sub = self._node.create_subscription(
            Image,
            "/camera/depth",
            self._depth_image_callback,
            10,
        )
        
        self.cv_bridge = CvBridge()
        
        self.tf2_buffer = Buffer()
        self.tf2_listener = TransformListener(self.tf2_buffer, self._node)
        
        self.experts = {
            "picking": PickingExpert(self._node, self.tf2_buffer),
            "placing": PlacingExpert(self._node, self.tf2_buffer)
        }
        
        self._node.get_logger().info("ExpertsNode initialized")


    def _depth_image_callback(self, msg: Image):
        cv_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        self.experts["picking"].update_depth_image(cv_image)
        self.experts["placing"].update_depth_image(cv_image)
        

def main(args=None):
    rclpy.init(args=args)
    node = rclpy.create_node("experts_node")
    experts_node = ExpertsNode(node)
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    rclpy.shutdown()
