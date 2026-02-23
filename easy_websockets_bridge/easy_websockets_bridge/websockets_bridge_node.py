import asyncio
import json
import websockets
from threading import Thread
import rclpy
from rclpy.node import Node
from easy_websockets_bridge.websockets_server import WebSocketServer  # Replace with your WS server lib
from easy_websockets_bridge.modules.bounding_box import BoundingBoxModule
from easy_websockets_bridge.modules.mode import ModeModule

class RosWebSocketBridge(Node):
    def __init__(self):
        super().__init__("easy_websockets_bridge")

        # -------------------------------
        # Event loop for async tasks
        # -------------------------------
        self.loop = asyncio.new_event_loop()
        t = Thread(target=self.loop.run_forever, daemon=True)
        t.start()

        # -------------------------------
        # WebSocket server for sending -> 9001
        # -------------------------------
        self.ws_send = WebSocketServer(port=9001)
        asyncio.run_coroutine_threadsafe(self.ws_send.start(), self.loop)

        # -------------------------------
        # WebSocket server for receiving -> 9002
        # -------------------------------
        self.ws_recv = WebSocketServer(port=9002)
        # Monkey-patch handler for receiving messages
        async def recv_handler(ws):
            self.ws_recv.clients.add(ws)
            try:
                async for msg in ws:
                    try:
                        data = json.loads(msg)
                        self._dispatch_ws_message(data)
                    except json.JSONDecodeError:
                        self.get_logger().warn(f"Invalid WS message: {msg}")
            except websockets.exceptions.ConnectionClosedOK:
                # Normal closure by client
                self.get_logger().info("WS client closed normally")
            except websockets.exceptions.ConnectionClosedError as e:
                # Unexpected closure
                self.get_logger().warn(f"WS client disconnected unexpectedly: {e}")
            finally:
                # Remove client and close properly if needed
                self.ws_recv.clients.remove(ws)
                if not ws.closed:
                    await ws.close(code=1000, reason="Server cleanup")

        self.ws_recv.handler = recv_handler
        asyncio.run_coroutine_threadsafe(self.ws_recv.start(), self.loop)

        # -------------------------------
        # Create modules
        # -------------------------------
        self.modules = {
            'bounding_box': BoundingBoxModule(self, self.ws_send),
            'mode': ModeModule(self, self.ws_send),
        }

        self.get_logger().info("ROS2 WebSocket bridge initialized (send:9001, recv:9002)")

    # -------------------------------
    # Send to clients safely
    # -------------------------------
    def send_ws(self, payload: dict):
        """Broadcast payload to all WS clients (9001)"""
        asyncio.run_coroutine_threadsafe(self.ws_send.broadcast(payload), self.loop)

    # -------------------------------
    # Dispatch inbound WS messages (9002)
    # -------------------------------
    def _dispatch_ws_message(self, data: dict):
        for module in self.modules.values():
            module.handle_ws_message(data)

def main():
    rclpy.init()
    node = RosWebSocketBridge()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
