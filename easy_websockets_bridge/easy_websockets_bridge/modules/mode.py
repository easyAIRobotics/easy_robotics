import asyncio
from easy_interfaces.msg import BoundingBoxes
from rclpy.node import Node

from easy_interfaces.srv import SetString
from std_msgs.msg import String

class ModeModule:
    def __init__(self, node: Node, ws_server):
        self._node = node
        self.ws = ws_server
        
        self.agent = None
        self.action = None
        
        self.agent_service = {}
        
        self.agent_subscription = {
            "skill_execution/action": self._node.create_subscription(
                String,
                "/skill_execution/action",
                self._on_skill_execution_action,
                10,
            ),
            "skill_execution/mode": self._node.create_subscription(
                String,
                "/skill_execution/mode",
                self._on_skill_execution_mode,
                10,
            ),
        }
        
        self._node.get_logger().info("ModeModule initialized")

    # Called later for inbound WS messages
    def handle_ws_message(self, data: dict):
        if data.get("type") == "selected_agent":
            self.handle_agent_selected(data)
            return
        
        if data.get("type") == "selected_action":
            self.handle_action_selected(data)
            return
        
        if data.get("type") == "set_mode":
            self.handle_set_mode(data)
            return
        
        if data.get("type") == "command":
            self.handle_command(data)
            return
        
    
    def handle_agent_selected(self, data: dict):
        self.agent = data["agent"]
        self._node.get_logger().info(
            f"Selected agent received: {self.agent}"
        )
            
        
    def handle_action_selected(self, data: dict):
        self.action = data["action"]
        self._node.get_logger().info(
            f"Selected action received: {self.action}"
        )
        
        # # Only call skill execution action selection
        # if self.agent != "skill_execution":
        #     return
        
        srv_name = self.agent + "/set_action"
        if not srv_name in self.agent_service:
            self._node.get_logger().info(
                f"Agent {srv_name} set action service not found, creating new one"
            )
            self.agent_service[srv_name] = self._node.create_client(
                srv_type=SetString,
                srv_name=srv_name,
            )
        else:
            self._node.get_logger().info(
                f"Agent {srv_name} set action service already exists"
            )
        
        # self.agent_service[srv_name].wait_for_service(timeout=2.0)
        req = SetString.Request()
        req.data = self.action
        self.agent_service[srv_name].call_async(req)
        
        return
        
        
    def handle_set_mode(self, data: dict):
        mode = data["mode"]
        self._node.get_logger().info(
            f"Set mode received: {mode}"
        )

        srv_name = self.agent + "/set_mode"

        if not srv_name in self.agent_service:
            self._node.get_logger().info(
                f"Agent {srv_name} set mode service not found, creating new one"
            )
            self.agent_service[srv_name] = self._node.create_client(
                srv_type=SetString,
                srv_name=srv_name,
            )
        else:
            self._node.get_logger().info(
                f"Agent {srv_name} set mode service already exists"
            )
            
        # self.agent_service[srv_name].wait_for_service(timeout=2.0)
        req = SetString.Request()
        req.data = mode
        self.agent_service[srv_name].call_async(req)
        
        return
    
    
    def handle_command(self, data: dict):
        command = data["command"]
        self._node.get_logger().info(
            f"Command received: {command}"
        )

        srv_name = self.agent + "/command"

        if not srv_name in self.agent_service:
            self._node.get_logger().info(
                f"Agent {srv_name} command service not found, creating new one"
            )
            self.agent_service[srv_name] = self._node.create_client(
                srv_type=SetString,
                srv_name=srv_name,
            )
        else:
            self._node.get_logger().info(
                f"Agent {srv_name} command service already exists"
            )
            
        # self.agent_service[srv_name].wait_for_service(timeout=2.0)
        req = SetString.Request()
        req.data = command
        self.agent_service[srv_name].call_async(req)
        
        return
    

    def _on_skill_execution_action(self, msg: String):
        payload = {
            "type": "skill_execution/action",
            "data": msg.data
        }
        
        self._node.send_ws(payload)
        
    def _on_skill_execution_mode(self, msg: String):
        payload = {
            "type": "skill_execution/mode",
            "data": msg.data
        }
        
        self._node.send_ws(payload)
