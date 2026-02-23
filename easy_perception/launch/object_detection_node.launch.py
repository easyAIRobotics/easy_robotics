import os
from ament_index_python import get_package_share_directory

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution

def generate_launch_description():
    model_path = os.path.join(
        get_package_share_directory('easy_perception'),
        'config',
        'box-sorting.pt'
    )

    object_detection_node = Node(
        package='easy_perception',
        executable='object_detection_node',
        name='easy_object_detection_node',
        parameters=[
            {"model_path": model_path},
        ],
        remappings=[
            ('camera/color', '/camera/image'),
            ('camera/depth', '/camera/depth_image'),
        ],
        output='screen'
    )

    return LaunchDescription([
        object_detection_node,
    ])