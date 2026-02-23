from moveit_configs_utils import MoveItConfigsBuilder
from moveit_configs_utils.launches import generate_demo_launch


def generate_launch_description():
    moveit_config = MoveItConfigsBuilder("kinova_g3_lite_box_picker", package_name="easy_kinova_g3l_moveit_config").to_moveit_configs()
    return generate_demo_launch(moveit_config)
