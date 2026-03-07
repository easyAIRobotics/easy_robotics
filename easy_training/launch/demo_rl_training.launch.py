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
            default_value='/home/hoang-dung/msc_hoang_dung_dinh/easy_ros_ws/datasets/skill_execution',
            description='Path to store training data (images, robot states, etc.)'
        ),
    ]
    
    skill_execution_trainer_node = Node(
        package='easy_training',
        executable='demo_rl_trainer',
        name='skill_execution_trainer_node',
        parameters=[
            {"agent_name": "skill_execution"},
            {"agent_type": "SkillExecution"},
            {"storage_path": LaunchConfiguration('storage_path')},
        ],
        remappings=[
            ('camera/depth', '/camera/depth_image'),
            ('camera/info', '/camera/camera_info'),
        ],
        output='screen'
    )
    
    experts_node = Node(
        package='easy_training',
        executable='experts_node',
        name='experts_node',
        remappings=[
            ('camera/depth', '/camera/depth_image'),
            ('camera/info', '/camera/camera_info'),
        ],
        output='screen'
    )
    
    return LaunchDescription(launch_args + [
        skill_execution_trainer_node,
        experts_node,
    ])