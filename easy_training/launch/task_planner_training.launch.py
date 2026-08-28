import os
from ament_index_python import get_package_share_directory

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    launch_args = [
        DeclareLaunchArgument(
            'storage_path',
            default_value='/home/hoang-dung/msc_hoang_dung_dinh/easy_ros_ws/datasets/task_planning',
            description='Path to store training data (images, robot states, etc.)'
        ),
    ]
    
    task_planning_trainer_node = Node(
        package='easy_training',
        executable='demo_rl_trainer',
        name='task_planning_trainer_node',
        parameters=[
            {"agent_name": "task_planning"},
            {"agent_type": "TaskPlanning"},
            {"storage_path": LaunchConfiguration('storage_path')},
        ],
        remappings=[
            ('camera/depth', '/camera/depth_image'),
            ('camera/rgb', '/camera/image'),
            ('camera/rgb_hand', '/camera_hand/image'),
            ('camera/info', '/camera/camera_info'),
        ],
        output='screen'
    )

    
    return LaunchDescription(launch_args + [
        task_planning_trainer_node,
    ])