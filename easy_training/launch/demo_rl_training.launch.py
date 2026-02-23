import os
from ament_index_python import get_package_share_directory

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

from moveit_configs_utils import MoveItConfigsBuilder

def generate_launch_description():
    # launch_arguments = {
    #     "use_fake_hardware": 'false',
    #     "gripper": "",
    #     "gripper_joint_name": "right_finger_bottom_joint",
    #     "dof": "6",
    # }

    # moveit_config = (
    #     MoveItConfigsBuilder(
    #         "kinova_g3_lite_box_picker", package_name="easy_kinova_g3l_moveit_config"
    #     )
    #     .robot_description(mappings=launch_arguments)
    #     .planning_scene_monitor(
    #         publish_robot_description=True, publish_robot_description_semantic=True
    #     )
    #     .planning_pipelines(pipelines=["ompl"])
    #     .to_moveit_configs()
    # )
    
    skill_execution_trainer_node = Node(
        package='easy_training',
        executable='demo_rl_trainer',
        name='skill_execution_trainer_node',
        parameters=[
            {"agent_name": "skill_execution"},
            {"agent_type": "SkillExecution"},
            # moveit_config.robot_description,
            # moveit_config.robot_description_semantic,
            # moveit_config.robot_description_kinematics,
            # moveit_config.planning_pipelines,
            # moveit_config.joint_limits,
        ],
        remappings=[
            ('camera/depth', '/camera/depth_image'),
            ('camera/info', '/camera/camera_info'),
        ],
        output='screen'
    )
    
    return LaunchDescription([
        skill_execution_trainer_node,
    ])