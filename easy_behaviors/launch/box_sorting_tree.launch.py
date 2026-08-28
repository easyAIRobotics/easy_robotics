#!/usr/bin/env python3

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
import os
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from datetime import datetime


def generate_launch_description():
    """
    Generate the launch description for the VR Behavior Tree application.

    Returns:
        LaunchDescription: Configured launch description with nodes and arguments.
    """

    pkg_easy_behaviors = get_package_share_directory("easy_behaviors")
    default_tree_path = os.path.join(pkg_easy_behaviors, "bt_xml", "box_sorting_skill_test.xml")

    # Declare launch argument for tree directory
    tree_path_arg = DeclareLaunchArgument(
        "tree_path",
        default_value=default_tree_path,
        description="Path to the behavior tree XML file (e.g., box_sorting_random_training.xml)",
    )

    # Use launch configuration
    tree_path = LaunchConfiguration("tree_path")

    # Define autonomy node with action server
    bt_run_node = Node(
        package="easy_behaviors",
        executable="box_sorting_tree_server",
        name="box_sorting_tree_server",
        output="screen",
        emulate_tty=True,
        parameters=[{"tree_path": tree_path}],
        # prefix=['xterm -e gdb -ex run --args'],
    )

    return LaunchDescription([tree_path_arg, bt_run_node])
