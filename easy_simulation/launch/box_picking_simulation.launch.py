from launch import LaunchDescription
from launch.actions import (
    SetEnvironmentVariable,
    DeclareLaunchArgument, IncludeLaunchDescription, ExecuteProcess, OpaqueFunction
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch.substitutions import Command, LaunchConfiguration, PathJoinSubstitution, FindExecutable

import os
from ament_index_python.packages import get_package_share_directory


def world_launch_setup(context, *args, **kwargs):
    world_name = LaunchConfiguration("world_name").perform(context)
    world_file_xacro = os.path.join(
        get_package_share_directory("easy_simulation"), "worlds", f"{world_name}.sdf.xacro"
    )
    
    world_file_sdf = os.path.join(
        get_package_share_directory("easy_simulation"), "worlds", f"{world_name}.sdf"
    )
    
    gz_config_file = os.path.join(
        get_package_share_directory("easy_simulation"), "config", "gz_config.config"
    )
    
    # Process the xacro file to generate the sdf file
    xacro_command = ExecuteProcess(
        cmd=[
            "ros2",
            "run",
            "xacro",
            "xacro",
            world_file_xacro,
            "-o",
            world_file_sdf,
        ],
        output="screen",
    )
    
    # ------------------
    # Launch Gazebo
    # ------------------
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(
                get_package_share_directory("ros_gz_sim"), "launch", "gz_sim.launch.py"
            )
        ]),
        launch_arguments={
            "gz_args": [" -r -v 4 -s ", world_file_sdf],
            # "gz_args": f" -r 4 --gui-config {gz_config_file} {world_file_sdf}",
        }.items(),
    )
    
    gazebo_gui = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(
                get_package_share_directory("ros_gz_sim"), "launch", "gz_sim.launch.py"
            )
        ]),
        launch_arguments={
            "gz_args": [" -r -v 4 -g ", gz_config_file],
            # "gz_args": f" -r 4 --gui-config {gz_config_file} {world_file_sdf}",
        }.items(),
    )
    
    # gz_server = IncludeLaunchDescription(
    #     PythonLaunchDescriptionSource([
    #         os.path.join(
    #             get_package_share_directory("ros_gz_sim"), "launch", "gz_server.launch.py"
    #         )
    #     ]),
    #     launch_arguments={
    #         "world_sdf_file": world_file_sdf,
    #     }.items(),
    # )

    return [
        xacro_command, 
        gazebo, 
        gazebo_gui,
    ]

def generate_launch_description():

    # ------------------
    # Launch Arguments
    # ------------------
    use_sim_time = DeclareLaunchArgument(
        "use_sim_time",
        default_value="true",
        description="Use simulation time",
    )
    
    world_name = DeclareLaunchArgument(
        "world_name",
        default_value="box_picking_desk",
        description="World file name to load",
    )
    
    # ------------------
    # Set gazebo model path
    # ------------------
    gazebo_models_path = SetEnvironmentVariable(
        'GZ_SIM_RESOURCE_PATH',
        PathJoinSubstitution([get_package_share_directory("easy_simulation"), 'worlds', 'models'])
    )

    robot_description_file = os.path.join(
        get_package_share_directory("easy_simulation"), "urdf", "kinova_g3l_box_picker.urdf.xacro"
    )

    # ------------------
    # Load Robot Description
    # ------------------
    robot_description_content = Command([
        FindExecutable(name="xacro"),
        " ",
        robot_description_file,
        " name:=kinova_arm"
    ])

    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="screen",
        parameters=[
            {"robot_description": robot_description_content},
            {"use_sim_time": LaunchConfiguration("use_sim_time")}
        ],
    )

    # ------------------
    # Spawn Robot in Gazebo
    # ------------------
    spawn_entity = Node(
        package="ros_gz_sim",
        executable="create",
        output="screen",
        arguments=[
            '-topic', 'robot_description',
            "-name", "kinova_g3_lite_box_picker",
        ],
    )
    
    base_link_contact_topics = 'world/empty/model/kinova_g3_lite_box_picker/link/base_link/sensor/base_link_contact_sensor/contact'
    arm_link_contact_topics = 'world/empty/model/kinova_g3_lite_box_picker/link/arm_link/sensor/arm_link_contact_sensor/contact'
    # end_effector_link_contact_topics = 'world/empty/model/kinova_g3_lite_box_picker/link/end_effector_link/sensor/suction_contact_sensor/contact'
    forearm_link_contact_topics = 'world/empty/model/kinova_g3_lite_box_picker/link/forearm_link/sensor/forearm_link_contact_sensor/contact'
    lower_wrist_link_contact_topics = 'world/empty/model/kinova_g3_lite_box_picker/link/lower_wrist_link/sensor/lower_wrist_link_contact_sensor/contact'
    shoulder_link_contact_topics = 'world/empty/model/kinova_g3_lite_box_picker/link/shoulder_link/sensor/shoulder_link_contact_sensor/contact'
    upper_wrist_link_contact_topics = 'world/empty/model/kinova_g3_lite_box_picker/link/upper_wrist_link/sensor/upper_wrist_link_contact_sensor/contact'

    ros_gz_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        name=f'ros_gz_bridge',
        parameters= [{
            'use_sim_time': LaunchConfiguration('use_sim_time'),
        }],
        arguments=[
            f'/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock',
            f'joint_states@sensor_msgs/msg/JointState[gz.msgs.Model',
            f'/camera/image@sensor_msgs/msg/Image[gz.msgs.Image',
            f'/camera/depth_image@sensor_msgs/msg/Image[gz.msgs.Image',
            f'/camera/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo',
            f'/cmd_suction@std_msgs/msg/Bool]gz.msgs.Boolean',
            f'/suction_state@std_msgs/msg/Bool[gz.msgs.Boolean',
            f'/cmd_suction_state@std_msgs/msg/Bool[gz.msgs.Boolean',
            f'/{base_link_contact_topics}@ros_gz_interfaces/msg/Contacts[gz.msgs.Contacts',
            f'/{arm_link_contact_topics}@ros_gz_interfaces/msg/Contacts[gz.msgs.Contacts',
            # f'/{end_effector_link_contact_topics}@ros_gz_interfaces/msg/Contacts[gz.msgs.Contacts',
            f'/{forearm_link_contact_topics}@ros_gz_interfaces/msg/Contacts[gz.msgs.Contacts',
            f'/{lower_wrist_link_contact_topics}@ros_gz_interfaces/msg/Contacts[gz.msgs.Contacts',
            f'/{shoulder_link_contact_topics}@ros_gz_interfaces/msg/Contacts[gz.msgs.Contacts',
            f'/{upper_wrist_link_contact_topics}@ros_gz_interfaces/msg/Contacts[gz.msgs.Contacts',
        ],
    )

    return LaunchDescription([
        use_sim_time,
        world_name,
        gazebo_models_path,
        robot_state_publisher,
        OpaqueFunction(function=world_launch_setup),
        spawn_entity,
        ros_gz_bridge,
    ])
