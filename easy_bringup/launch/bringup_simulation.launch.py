from launch import LaunchDescription
from launch.actions import (
    IncludeLaunchDescription
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

import os
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    easy_simulation = get_package_share_directory("easy_simulation")
    easy_kinova_g3l_moveit_config = get_package_share_directory("easy_kinova_g3l_moveit_config")
    easy_websockets_bridge = get_package_share_directory("easy_websockets_bridge")
    
    """
    Launch params
    """
    
    """
    Simulation
    """
    simulation_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(easy_simulation, "launch", "box_picking_simulation.launch.py")
        ),
    )
    
    """
    MoveIt2 for Kinova Gen3 Lite
    """
    moveit_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(easy_kinova_g3l_moveit_config, "launch", "robot_sim.launch.py")
        ),
    )
    
    """
    Box spawner node
    """
    box_spawner_node = Node(
        package="easy_simulation",
        executable="box_spawner.py",
        name="box_spawner_node",
        output="screen",
    )
    
    """
    Web interfaces
    """
    easy_websockets_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                easy_websockets_bridge, "launch", "websockets_bridge.launch.py")
        ),
    )
    
    return LaunchDescription([
        simulation_launch,
        moveit_launch,
        box_spawner_node,
        # easy_websockets_launch,
    ])