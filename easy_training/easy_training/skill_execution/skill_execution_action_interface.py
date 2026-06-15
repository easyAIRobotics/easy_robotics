import time

import cv2
import rclpy
from rclpy.node import Node
from easy_training.agent_interfaces import ActionInterface, StateInterface, AgentMode
from easy_training.utils import *

from moveit.planning import MoveItPy
from moveit.core.robot_state import RobotState
from moveit.core.planning_scene import PlanningScene

from std_msgs.msg import Float64MultiArray, Bool
from std_srvs.srv import SetBool
from easy_interfaces.srv import SolveIK, CheckCollision
from controller_manager_msgs.srv import SwitchController
from geometry_msgs.msg import TransformStamped

from tf2_ros import TransformBroadcaster

import copy

IK_FAILURE_PENALTY = 2.0
COLLISION_PENALTY = 5.0
EEF_MOVEMENT_PENALTY_SCALE = 1.0

MOVEGROUP_NAME = 'suction_tip'
BASE_FRAME = 'base_link'

MAX_JOINT_DELTA = 0.1  # Maximum allowed joint position change

JOINT_LOWER_LIMITS = np.array([-2.18, -1.48, -0.58, -2.58, -2.51, -2.58])
JOINT_UPPER_LIMITS = np.array([2.18, 0.5, 2.59, 2.58, 2.51, 2.58])

ACTION_MODE = "joint_positions" # or "eef_pose"
# ACTION_MODE = "eef_pose"

def check(name, x):
    # print type if not a numpy array
    if not isinstance(x, np.ndarray):
        print(f"[{name}] Warning: Expected a numpy array but got {type(x)}", flush=True)
        return False
    # Check nan values of a numpy array and print the name and values if any are found
    if np.isnan(x).any():
        print(f"[{name}] NaN values found: {x}", flush=True)
        return False
    return True

class SkillExecutionActionInterface(ActionInterface):
    def __init__(self, node: Node):
        super().__init__(node)
        
        self._mode = AgentMode.SELF_LEARNING
        self.last_state = None
        
        self.solve_ik_client = self._node.create_client(
            SolveIK, 'solve_ik', callback_group=self.action_interface_callback_group)
        self.check_collision_client = self._node.create_client(
            CheckCollision, 'check_collision', callback_group=self.action_interface_callback_group)
        
        self.switch_controller_client = self._node.create_client(
            SwitchController, 'controller_manager/switch_controller', callback_group=self.action_interface_callback_group)
        
        self.enable_picking_expert_client = self._node.create_client(
            SetBool, "picking_expert/enable", callback_group=self.action_interface_callback_group)
        
        self.enable_placing_expert_client = self._node.create_client(
            SetBool, "placing_expert/enable", callback_group=self.action_interface_callback_group)
        
        self.joint_command_pub = self._node.create_publisher(
            Float64MultiArray, 'forward_position_joint_controller/commands', 1, callback_group=self.action_interface_callback_group)
        
        self.cmd_suction_pub = self._node.create_publisher(
            Bool, 'cmd_suction', 1, callback_group=self.action_interface_callback_group)
        
        self.tf_broadcaster = TransformBroadcaster(self._node)
        
        
    def set_action(self, action):
        if self.action == action:
            return
        super().set_action(action)
        # Enable or disable expert based on mode
        enable_expert_req = SetBool.Request()
        enable_expert_req.data = (self._mode == AgentMode.BEHAVIOR_CLONING)
        
        if self.action == "pick":
            self.enable_picking_expert_client.call_async(enable_expert_req)
        elif self.action == "place":
            self.enable_placing_expert_client.call_async(enable_expert_req)
            
        # Disable all other experts to avoid conflicts
        if self.action != "pick":
            disable_picking_req = SetBool.Request()
            disable_picking_req.data = False
            self.enable_picking_expert_client.call_async(disable_picking_req)
        if self.action != "place":
            disable_placing_req = SetBool.Request()
            disable_placing_req.data = False
            self.enable_placing_expert_client.call_async(disable_placing_req)
        
        
    def set_mode(self, mode: AgentMode):
        if self._mode == mode:
            return
        super().set_mode(mode)
        self._node.get_logger().info(f"[SkillExecutionActionInterface] Mode set to: {self._mode.name}")
        
        # Switch controllers based on mode
        sw_req = SwitchController.Request()
        if self._mode == AgentMode.BEHAVIOR_CLONING or self._mode == AgentMode.IDLE:
            sw_req.deactivate_controllers = ['forward_position_joint_controller']
            sw_req.activate_controllers = ['joint_trajectory_controller']
        else:
            sw_req.deactivate_controllers = ['joint_trajectory_controller']
            sw_req.activate_controllers = ['forward_position_joint_controller']
        sw_req.strictness = SwitchController.Request.STRICT
        sw_req.activate_asap = True

        # Enable or disable expert based on mode
        enable_expert_req = SetBool.Request()
        enable_expert_req.data = (self._mode == AgentMode.BEHAVIOR_CLONING)
        
        if self.action == "pick":
            self.enable_picking_expert_client.call_async(enable_expert_req)
        elif self.action == "place":
            self.enable_placing_expert_client.call_async(enable_expert_req)
            
        future = self.switch_controller_client.call_async(sw_req)
        while not future.done():
            time.sleep(0.001)
        if future.result() is not None:
            res = future.result()
            if res.ok:
                self._node.get_logger().info(f"[SkillExecutionActionInterface] Controller switch successful")
            else:
                self._node.get_logger().error(f"[SkillExecutionActionInterface] Controller switch failed")
                
    
    # Perform action and return the reward"""
    def perform(self, act_vec: list, state_interface: StateInterface) -> dict:
        if self.last_state is None:
            self.last_state = copy.deepcopy(state_interface.get_state())  # Get current state
        
        transition = {
            "image": state_interface.get_image(),
            "target_image": state_interface.get_target_image(),
            "original_target_image": state_interface.get_original_target_image(),
            "skill": state_interface.get_skill(),
            "robot_state": state_interface.get_robot_state(),
        }
        
        reward = 0.0
        if act_vec:
            current_state = state_interface.get_state()
            if ACTION_MODE == "eef_pose":
                act = {
                    "eef_pose": act_vec[:9],
                    "suction_command": act_vec[9]
                }
                IK_success, joint_positions = self.solve_IK(act["eef_pose"], current_state["joint_positions"])
                self._broadcast_target_tf(act["eef_pose"])
            else:  # ACTION_MODE == "joint_positions"
                act = {
                    "joint_positions": act_vec[:6],
                    "suction_command": act_vec[6]
                }
                joint_positions = act["joint_positions"]
                IK_success = True
            bounded_joint_positions, penalty = self._make_bounded_joint_positions(current_state["joint_positions"], joint_positions, MAX_JOINT_DELTA)
                   
            if bounded_joint_positions and not self._check_joint_limits(bounded_joint_positions):
                print(f"[SkillExecutionActionInterface] Joint limits violated for positions: {bounded_joint_positions}", flush=True)
                bounded_joint_positions = [
                    np.clip(bounded_joint_positions[i], JOINT_LOWER_LIMITS[i], JOINT_UPPER_LIMITS[i]) for i in range(len(bounded_joint_positions))
                ]
                reward -= IK_FAILURE_PENALTY
            
            if bounded_joint_positions:
                if not self.check_collision(bounded_joint_positions):
                    self.send_joint_command(bounded_joint_positions)
                else:
                    reward -= COLLISION_PENALTY
                self.send_gripper_command(act["suction_command"])
            taken_act = act
            
            self.wait_for_next_state()
            
            if not IK_success:
                reward -= IK_FAILURE_PENALTY  # Penalize for IK failure
            
            if state_interface.check_collision():
                self._node.get_logger().warn(f"[SkillExecutionActionInterface] Penalize collision action in active mode")
                reward -= COLLISION_PENALTY  # Penalize for collision
            
            if ACTION_MODE == "eef_pose":
                taken_act["joint_positions"] = joint_positions
            else:
                taken_act["eef_pose"] = state_interface.get_state()["eef_pose"]
            
            reward += state_interface.get_reward() 
            act_done = state_interface.get_done()
                
        if not act_vec:
            if self._mode == AgentMode.BEHAVIOR_CLONING:
                self.wait_for_next_state()
                next_state = state_interface.get_state()  # Get current state
                taken_act = {
                    "eef_pose": next_state["eef_pose"],
                    "joint_positions": next_state["joint_positions"],
                    "suction_command": next_state["cmd_suction_state"]
                }
                
                # Dont record expert demonstration if no movement is taken
                tf_dist = joint_distance(self.last_state["joint_positions"], next_state["joint_positions"])
                reward += state_interface.get_reward()
                act_done = state_interface.get_done()
                
                self._node.get_logger().info(f"suction_command: {taken_act['suction_command']} vs last cmd {self.last_state['cmd_suction_state']}, tf_dist: {tf_dist}")
                if tf_dist < 0.1  and act_done == 0.0 and \
                        taken_act["suction_command"] == self.last_state["cmd_suction_state"]:
                    return 0.0, {}  # No action taken, skipping
                
                if tf_dist > 0.5 and act_done == 0.0 and \
                        taken_act["suction_command"] == self.last_state["cmd_suction_state"]:
                    self.last_state = copy.deepcopy(state_interface.get_state())
                    return 0.0, {}  # Large movement without suction change, likely not a valid demo, skipping                        
            
                if state_interface.check_collision():
                    self._node.get_logger().warn(f"[SkillExecutionActionInterface] Penalize collision action in passive mode")
                    reward -= COLLISION_PENALTY  # Penalize for collision
                    
                if act_done == 1.0:
                    time.sleep(1.0)  # Extra wait
                         
            # taken_act = do_reverse_transform(taken_act_eef_pose["eef_pose"], state["eef_pose"]) + [taken_act_eef_pose["suction_command"]]
        self.last_state = copy.deepcopy(state_interface.get_state())
        taken_j_action = taken_act["joint_positions"] + [taken_act["suction_command"]]
        taken_e_action = taken_act["eef_pose"] + [taken_act["suction_command"]]

        print(f"DONEEEEEE: {act_done}", flush=True)
        transition.update({
            "j_action": np.array(taken_j_action),
            "e_action": np.array(taken_e_action),
            "next_image": state_interface.get_image(),
            "next_target_image": state_interface.get_target_image(),
            "next_original_target_image": state_interface.get_original_target_image(),
            "next_skill": state_interface.get_skill(),
            "next_robot_state": state_interface.get_robot_state(),
            "done": act_done
        })
        # check("j_action", transition["j_action"])
        # check("e_action", transition["e_action"])
        # check("next_image", transition["next_image"])
        # check("next_target_image", transition["next_target_image"])
        # check("next_original_target_image", transition["next_original_target_image"])
        # check("next_robot_state", transition["next_robot_state"])
        
        if not (transition["next_original_target_image"] is None or transition["original_target_image"] is None):
            return reward, transition
        else:
            return 0.0, {}
    
    
    def solve_IK(self, target_pose: list[float], initial_joint_positions: list[float]) -> list[float]:
        target_pose_quat = rot6d_to_quaternion(target_pose[3:9])
        
        req = SolveIK.Request()
        req.group_name = MOVEGROUP_NAME
        req.target_pose = target_pose[:3] + target_pose_quat
        req.initial_joint_positions = initial_joint_positions
        
        future = self.solve_ik_client.call_async(req)
        while not future.done():
            time.sleep(0.001)
        if future.result() is not None:
            res = future.result()
            print(f"[SkillExecutionActionInterface] IK service response: success={res.success}, joints='{res.joint_positions}'", flush=True)
                
        return res.success, res.joint_positions.tolist()
    
    
    def check_collision(self, joint_positions: list[float]) -> bool:
        req = CheckCollision.Request()
        req.group_name = MOVEGROUP_NAME
        req.joint_positions = joint_positions
        
        self.check_collision_client.wait_for_service()
        future = self.check_collision_client.call_async(req)
        while not future.done():
            time.sleep(0.001)
        if future.result() is not None:
            res = future.result()
            collision = res.in_collision
            print(f"[SkillExecutionActionInterface] Collision check result: {'In collision' if collision else 'No collision'}", flush=True)
        else:
            print(f"[SkillExecutionActionInterface] Collision check service call failed", flush=True)
            collision = True  # Assume collision if service call fails
            
        return collision
    
    
    def send_joint_command(self, joint_positions: list[float]):
        joint_command_msg = Float64MultiArray()
        joint_command_msg.data = joint_positions
        self.joint_command_pub.publish(joint_command_msg)
        
        
    def send_gripper_command(self, suction_command: float):
        suction_on = suction_command > 0.5
        suction_command_msg = Bool()
        suction_command_msg.data = suction_on
        self.cmd_suction_pub.publish(suction_command_msg)
        
    
    def _broadcast_target_tf(self, target_pose: list[float]):
        target_pose_quat = rot6d_to_quaternion(target_pose[3:9])
        t = TransformStamped()
        t.header.stamp = self._node.get_clock().now().to_msg()
        t.header.frame_id = BASE_FRAME
        t.child_frame_id = "target_eef_pose"
        t.transform.translation.x = target_pose[0]
        t.transform.translation.y = target_pose[1]
        t.transform.translation.z = target_pose[2]
        t.transform.rotation.x = target_pose_quat[0]
        t.transform.rotation.y = target_pose_quat[1]
        t.transform.rotation.z = target_pose_quat[2]
        t.transform.rotation.w = target_pose_quat[3]
        
        self.tf_broadcaster.sendTransform(t)

    def _make_bounded_joint_positions(self, current_joints: list[float], target_joints: list[float], max_delta: float) -> list[float]:
        bounded_joints = []
        j_delta = np.array(target_joints) - np.array(current_joints)
        abs_delta = np.abs(j_delta)
        penalty = abs_delta[abs_delta >= MAX_JOINT_DELTA].sum()
        max_abs_delta = np.max(np.abs(j_delta))
        print(f"max abs delta: {max_abs_delta}, j_delta: {j_delta}", flush=True)
        if max_abs_delta > max_delta:
            j_delta = (j_delta / max_abs_delta) * max_delta
        bounded_joints = np.array(current_joints) + j_delta
        return bounded_joints.tolist(), penalty
    
    
    def _check_joint_limits(self, joint_positions: list[float]) -> bool:
        for i, joint in enumerate(joint_positions):
            if joint < JOINT_LOWER_LIMITS[i] or joint > JOINT_UPPER_LIMITS[i]:
                return False
        return True
    