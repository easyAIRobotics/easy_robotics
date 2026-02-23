import os
from ament_index_python import get_package_share_directory

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution

def generate_launch_description():
    port_arg = DeclareLaunchArgument(
        "port",
        default_value="9001",
        description="Port for the WebSocket server",
    )

    websockets_bridge_node = Node(
        package="easy_websockets_bridge",
        executable="websockets_bridge_node",
        name="easy_websockets_bridge_node",
        output="screen",
        parameters=[
            {
                "port": LaunchConfiguration("port"),
            }
        ],
        remappings=[
            ('/detections', '/easy_object_detection/bounding_boxes'),
        ],
    )

    return LaunchDescription([
        port_arg,
        websockets_bridge_node,
    ])