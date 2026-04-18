#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger

import numpy as np
import time
import gz.msgs10.entity_pb2 as entity_msgs
import gz.msgs10.entity_factory_pb2 as factory_msgs
import gz.msgs10.pose_v_pb2 as pose_v_msgs
import gz.msgs10.boolean_pb2 as bool_msgs
import gz.transport13 as gz_transport

from tf_transformations import quaternion_from_euler
import threading

NUM_BOXES = 3
NUM_TARGET_BOXES = 3
AREA_RADIUS = 0.35
WORLD_NAME = "empty"     # Change if using another world name
SDF_TEMPLATE = """
    <?xml version="1.0" ?>
    <sdf version="1.6">
        <include>
        <uri>{}</uri>
        </include>
    </sdf>
"""
def get_sdf_model(name):
    return SDF_TEMPLATE.format(name)

class BoxSpawner:
    def __init__(self):
        self.ros_node = Node("box_spawner")
        self.gz_node = gz_transport.Node()

        self.mutex = threading.Lock()

        # Subscribe to model poses to know existing models
        self.models = set()
        self.gz_node.subscribe(
            pose_v_msgs.Pose_V,
            f"/world/{WORLD_NAME}/pose/info",
            self.pose_callback
        )

        # Box model URIs
        self.box_uri = {
            "box": "model://cardboard_box",
            "target_box": "model://cardboard_box_green",
            "storage_bin": "model://storage_bin",
            "storage_bin_green": "model://storage_bin_green",
        }
        
        self.storage_bin_positions = {
            "storage_bin": [0.0, 0.5, 1.15],
            "storage_bin_green": [0.0, -0.5, 1.15],
        }

        self.center_pose = [0.3, 0.0, 1.6]
        self.std_pose = [AREA_RADIUS / 3, AREA_RADIUS / 3, 0.0]
        
        self.respawn_boxes_srv = self.ros_node.create_service(
            Trigger,
            "respawn_boxes",
            self.respawn_boxes_callback
        )
        self.ros_node.get_logger().info("[BoxSpawner] Initialized using Gazebo Transport.")

    # --------------------------------------------------------------

    def respawn_boxes_callback(self, request, response):
        # Wait for model poses to be available
        while not self.models:
            time.sleep(0.1)
        with self.mutex:
            self.respawn_boxes()
        self.models = {}
        response.success = True
        response.message = "Boxes respawned."
        return response    
    
    def pose_callback(self, msg: pose_v_msgs.Pose_V):
        """Update current model names."""
        with self.mutex:
            self.models = {p.id: p.name for p in msg.pose}

    # --------------------------------------------------------------

    def spawn_box(self, name, type, pose):
        """Spawn model via Ignition transport."""
        req = factory_msgs.EntityFactory()
        req.name = name
        req.sdf = get_sdf_model(self.box_uri[type])
        # Set pose
        req.pose.position.x = pose[0]
        req.pose.position.y = pose[1]
        req.pose.position.z = pose[2]

        # qx, qy, qz, qw = quaternion_from_euler(pose[3], pose[4], pose[5])

        # req.pose.orientation.x = qx
        # req.pose.orientation.y = qy
        # req.pose.orientation.z = qz
        # req.pose.orientation.w = qw

        # self.spawn_client.call(req)
        res, resp = self.gz_node.request(
            f"/world/{WORLD_NAME}/create",
            req,
            request_type=factory_msgs.EntityFactory,
            response_type=bool_msgs.Boolean,
            timeout=1000
        )

        if res and resp:
            self.ros_node.get_logger().info(f"[BoxSpawner] Spawned {name} at {[pose[0], pose[1], pose[2]]}, response: {resp}")
        else:
            self.ros_node.get_logger().info(f"[BoxSpawner] Failed to spawn {name}, response: {resp}")

        time.sleep(0.1)  # allow some time for spawning

    # --------------------------------------------------------------

    def delete_box(self, id, name):
        req = entity_msgs.Entity()
        req.name = name
        req.id = id
        req.type = entity_msgs.Entity.MODEL
        resp = bool_msgs.Boolean()
        # self.delete_client.call(req)
        res, resp = self.gz_node.request(
            f"/world/{WORLD_NAME}/remove",
            req,
            request_type=entity_msgs.Entity,
            response_type=bool_msgs.Boolean,
            timeout=1
        )
        self.ros_node.get_logger().info(f"[BoxSpawner] Deleted: {name}")

    # --------------------------------------------------------------

    def respawn_boxes(self):
        self.ros_node.get_logger().info("[BoxSpawner] Clearing boxes...")
        for id in self.models:
            if ("box" in self.models[id] and len(self.models[id]) < 7) or "target_box" in self.models[id]:
                self.delete_box(id, self.models[id])
            if "storage_bin" in self.models[id]:
                self.delete_box(id, self.models[id])

        time.sleep(1.0)  # allow deletion to propagate
        
        for storage_bin_name in self.storage_bin_positions.keys():    
            self.spawn_box(storage_bin_name, storage_bin_name, self.storage_bin_positions[storage_bin_name] + [0.0, 0.0, 0.0])

        num_spawned = {'box': 0, 'target_box': 0}
        id = 0

        while num_spawned['box'] < NUM_BOXES or num_spawned['target_box'] < NUM_TARGET_BOXES:
            if np.random.rand() < (NUM_TARGET_BOXES - num_spawned['target_box']) / \
                    (NUM_BOXES + NUM_TARGET_BOXES - sum(num_spawned.values())):
                box_type = 'target_box'
            else:
                box_type = 'box'

            box_name = f"{box_type}_{id}"

            x = np.random.normal(self.center_pose[0], self.std_pose[0])
            y = np.random.normal(self.center_pose[1], self.std_pose[1])
            z = self.center_pose[2]

            roll = np.random.uniform(-np.pi, np.pi)
            pitch = np.random.uniform(-np.pi, np.pi)
            yaw = np.random.uniform(-np.pi, np.pi)

            self.spawn_box(box_name, box_type, [x, y, z, roll, pitch, yaw])
            num_spawned[box_type] += 1
            id += 1
            

        self.ros_node.get_logger().info("[BoxSpawner] Done.")

    # --------------------------------------------------------------

    def run(self):
        self.ros_node.get_logger().info("[BoxSpawner] Running loop. Press CTRL+C to exit.")
        while True:
            rclpy.spin_once(self.ros_node, timeout_sec=0.1)
            time.sleep(0.01)

# --------------------------------------------------------------

def main():
    rclpy.init()
    spawner = BoxSpawner()
    spawner.run()

if __name__ == "__main__":
    main()
