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

IK_FAILURE_PENALTY = 0.0
COLLISION_PENALTY = 5.0
EEF_MOVEMENT_PENALTY_SCALE = 1.0

MOVEGROUP_NAME = 'suction_tip'
BASE_FRAME = 'base_link'

class SkillExecutionActionInterface(ActionInterface):
    def __init__(self, node: Node):
        super().__init__(node)
        
        self._mode = AgentMode.SELF_LEARNING
        
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
        super().set_mode(mode)
        self._node.get_logger().info(f"[SkillExecutionActionInterface] Mode set to: {self._mode.name}")
        
        # Switch controllers based on mode
        sw_req = SwitchController.Request()
        if self._mode == AgentMode.BEHAVIOR_CLONING:
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
        state = copy.deepcopy(state_interface.get_state())  # Get current state
        
        reward = 0.0
        if act_vec:
            act = {
                "joint_positions": act_vec[:6],
                "suction_command": act_vec[6]
            }
            
            # act_eef_pose = do_transform(act["eef_pose"], state["eef_pose"])                      
            # joint_positions = self.solve_IK(act["eef_pose"], state["joint_positions"])                
            
            # TODO: Verify collision before executing action
            # print(f"Current state joint positions: {state['joint_positions']} \n target joint positions: {joint_positions}", flush=True)
            # self._broadcast_target_tf(act["eef_pose"])
            # j_dist = joint_distance(state["joint_positions"], joint_positions) if joint_positions else float('inf')
            # print(f"Joint distance to target: {j_dist}", flush=True)
            # if joint_positions and j_dist < 0.5:
            #     self.send_joint_command(joint_positions)
            #     self.send_gripper_command(act["suction_command"])
            # else:
            #     reward -= IK_FAILURE_PENALTY  # Penalize for IK failure
            # target_joint_positions = np.array(state["joint_positions"]) + np.array(act["delta_joint_positions"])
            
            self.send_joint_command(act["joint_positions"])
            self.send_gripper_command(act["suction_command"])
            taken_act = act
            
            self.wait_for_next_state()
            if state_interface.check_collision():
                self._node.get_logger().warn(f"[SkillExecutionActionInterface] Penalize collision action in active mode")
                reward -= COLLISION_PENALTY  # Penalize for collision
                
        if not act_vec:
            if self._mode == AgentMode.BEHAVIOR_CLONING:
                self.wait_for_next_state()
                next_state = state_interface.get_state()  # Get current state
                taken_act = {
                    "joint_positions": next_state["joint_positions"],
                    "suction_command": next_state["cmd_suction_state"]
                }
                
                # Dont record expert demonstration if no movement is taken
                tf_dist = joint_distance(state["joint_positions"], next_state["joint_positions"])
                if tf_dist < 5e-3 and \
                        taken_act["suction_command"] == state["cmd_suction_state"]:
                    return 0.0, {}  # No action taken, skipping
                            
            
                if state_interface.check_collision():
                    self._node.get_logger().warn(f"[SkillExecutionActionInterface] Penalize collision action in passive mode")
                    reward -= COLLISION_PENALTY  # Penalize for collision
                    
        
            # taken_act = do_reverse_transform(taken_act_eef_pose["eef_pose"], state["eef_pose"]) + [taken_act_eef_pose["suction_command"]]
        
        taken_act_vec = taken_act["joint_positions"] + [taken_act["suction_command"]]

        # Accumulate reward from next state
        reward += state_interface.get_reward()
        print(f"DONEEEEEE: {state_interface.get_done()}", flush=True)
        transition = {
            "action": np.array(taken_act_vec),
            "image": state_interface.get_image(),
            "skill": state_interface.get_skill(),
            "robot_state": state_interface.get_robot_state(),
            "next_image": state_interface.get_image(),
            "next_skill": state_interface.get_skill(),
            "next_robot_state": state_interface.get_robot_state(),
            "done": state_interface.get_done()
        }
        
        return reward, transition
    
    
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
            if res.success:
                joint_positions = res.joint_positions
                print(f"[SkillExecutionActionInterface] IK solution found: {joint_positions}", flush=True)
            else:
                print(f"[SkillExecutionActionInterface] IK solution not found with reason: {res.message}", flush=True)
                joint_positions = []
                
        return joint_positions
    
    
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
        suction_on = suction_command > 0.0
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
