import os
from ament_index_python import get_package_share_directory

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution

def generate_launch_description():
    """ Parameters """
    
    launch_params = [
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='false',
            description='Use simulation time'
        ),
        DeclareLaunchArgument(
            'video_device',
            default_value='/dev/video0',
            description='Video device for the webcam'
        ),
        
        DeclareLaunchArgument(
            'hf_repo_id',
            default_value='/home/hoang-dung/easy_jazzy_ws/third-party/sam-3d-body-vith',
            description='Hugging Face repository ID for the model'
        ),
    ]

    """ Nodes """
    
    webcam_node = Node(
        package='usb_cam',
        executable='usb_cam_node_exe',
        parameters=[{
            'video_device': LaunchConfiguration('video_device'),
        }],
        output='screen'
    )
    
    human_pose_estimation_node = Node(
        package='easy_perception',
        executable='human_pose_estimation_node',
        name='human_pose_estimation_node',
        parameters=[{
            'use_sim_time': LaunchConfiguration('use_sim_time'),
            'hf_repo_id': LaunchConfiguration('hf_repo_id'),
            'device': 'cuda:0',
        }],
        remappings=[
            ('input_image', '/image_raw'),
        ],
        output='screen'
    )

    return LaunchDescription(launch_params + [
        webcam_node,
        human_pose_estimation_node,
    ])