from rclpy.node import Node

from ros_gz_interfaces.msg import Contacts

class ContactSensorHandler:
    def __init__(self, node: Node, name: str):
        self._node = node
        self.in_collision = False
        
        self._node.declare_parameter(f'{name}.topic', f'')
        topic_name = self._node.get_parameter(f'{name}.topic').get_parameter_value().string_value
        self._node.declare_parameter(f'{name}.exceptions', [""])
        self.exceptions = self._node.get_parameter(f'{name}.exceptions').get_parameter_value().string_array_value
        
        if not topic_name:
            self._node.get_logger().error(f"[ContactSensorHandler] No topic specified for contact sensor '{name}'")
            raise ValueError(f"No topic specified for contact sensor '{name}'")
        
        self._subscription = self._node.create_subscription(
            Contacts,
            topic_name,
            self._contact_callback,
            10
        )
        

    def _contact_callback(self, msg: Contacts):
        for contact in msg.contacts:
            if contact.collision1.name in self.exceptions or contact.collision2.name in self.exceptions:
                continue
            
            self.in_collision = True
            return
        
        self.in_collision = False


class CollisionMonitor:
    def __init__(self, node: Node):
        self._node = node
        self.contact_sensors = []
        
        self._node.declare_parameter("contact_sensors", [""])
        sensor_names = self._node.get_parameter("contact_sensors").get_parameter_value().string_array_value
        
        for name in sensor_names:
            try:
                sensor_handler = ContactSensorHandler(node, name)
                self.contact_sensors.append(sensor_handler)
                self._node.get_logger().info(f"[CollisionMonitor] Contact sensor '{name}' initialized successfully")
            except ValueError as e:
                self._node.get_logger().error(f"[CollisionMonitor] Failed to initialize contact sensor '{name}': {e}")
                
                
    def check_collisions(self) -> bool:
        for sensor in self.contact_sensors:
            if sensor.in_collision:
                return True
        return False