#!/usr/bin/env python3

import os
from pathlib import Path
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import time
import threading
import sys
import tty
import termios

class ImageSaverNode(Node):
    def __init__(self):
        super().__init__('image_saver')
        # Parameters: topic to subscribe, output folder, filename prefix
        self.declare_parameter('image_topic', '/camera/image')
        self.declare_parameter('output_dir', '/home/hoang-dung/easy_jazzy_ws/datasets/box-container-detection')
        self.declare_parameter('filename_prefix', 'image_')

        self.image_topic = self.get_parameter('image_topic').get_parameter_value().string_value
        out_dir_param = self.get_parameter('output_dir').get_parameter_value().string_value
        self.output_dir = Path(out_dir_param).expanduser().resolve()
        self.prefix = self.get_parameter('filename_prefix').get_parameter_value().string_value

        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)

        self.bridge = CvBridge()
        self.counter = 0

        # Store latest image (updated on each incoming message)
        self._last_image = None
        self._last_stamp = None
        self._lock = threading.Lock()
        self._stop_event = threading.Event()

        self.get_logger().info(f"Subscribing to: {self.image_topic}")
        self.get_logger().info(f"Saving images to: {self.output_dir}")
        self.get_logger().info("Press SPACE in this terminal to save the latest image. Press Ctrl-C to exit.")

        self.sub = self.create_subscription(
            Image,
            self.image_topic,
            self.image_callback,
            qos_profile_sensor_data
        )

        # Start keyboard thread to listen for space key
        self._kb_thread = threading.Thread(target=self._keyboard_listener, daemon=True)
        self._kb_thread.start()

    def image_callback(self, msg: Image):
        try:
            # Convert incoming message to OpenCV image and store it
            try:
                cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            except CvBridgeError:
                cv_image = self.bridge.imgmsg_to_cv2(msg)

            with self._lock:
                self._last_image = cv_image.copy() if cv_image is not None else None
                # store stamp if available
                self._last_stamp = getattr(msg, 'header', None).stamp if getattr(msg, 'header', None) else None

        except Exception as e:
            self.get_logger().error(f"Error processing image message: {e}")

    def _keyboard_listener(self):
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
            while rclpy.ok() and not self._stop_event.is_set():
                ch = sys.stdin.read(1)
                if not ch:
                    continue
                if ch == ' ':
                    self._save_latest_image()
                # allow quitting from keyboard thread (optional)
                if ch in ('q', '\x03'):  # 'q' or Ctrl-C
                    self.get_logger().info("Keyboard requested shutdown")
                    rclpy.shutdown()
                    break
        except Exception as e:
            # Do not crash the node from keyboard thread errors
            self.get_logger().error(f"Keyboard listener error: {e}")
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    def _save_latest_image(self):
        with self._lock:
            if self._last_image is None:
                self.get_logger().warning("No image received yet to save.")
                return
            cv_image = self._last_image.copy()
            stamp = self._last_stamp

        # Build filename using header timestamp if available, otherwise current time
        if stamp and getattr(stamp, 'sec', None) is not None:
            try:
                secs = int(stamp.sec)
                nsecs = int(getattr(stamp, 'nanosec', getattr(stamp, 'nsec', 0)))
                timestamp = f"{secs:010d}_{int(nsecs/1000):06d}"
            except Exception:
                t = time.time()
                timestamp = f"{int(t)}_{int((t-int(t))*1e6):06d}"
        else:
            t = time.time()
            timestamp = f"{int(t)}_{int((t-int(t))*1e6):06d}"

        filename = f"{self.prefix}{self.counter:06d}_{timestamp}.png"
        filepath = str(self.output_dir / filename)

        try:
            success = cv2.imwrite(filepath, cv_image)
            if not success:
                self.get_logger().error(f"Failed to write image to {filepath}")
            else:
                self.get_logger().info(f"Saved image: {filepath}")
                self.counter += 1
        except Exception as e:
            self.get_logger().error(f"Error saving image to disk: {e}")

    def stop(self):
        self._stop_event.set()
        if self._kb_thread.is_alive():
            try:
                # attempt to wake the blocking stdin read
                import os, signal
                os.kill(os.getpid(), signal.SIGINT)
            except Exception:
                pass
            self._kb_thread.join(timeout=1)


def main(args=None):
    rclpy.init(args=args)
    node = ImageSaverNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info("Shutting down image_saver node")
        node.stop()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
