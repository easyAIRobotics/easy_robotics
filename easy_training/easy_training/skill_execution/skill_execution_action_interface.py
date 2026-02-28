import time
import rclpy
from rclpy.node import Node
from easy_training.agent_interfaces import ActionInterface, StateInterface, AgentMode
from easy_training .utils import *

from moveit.planning import MoveItPy
from moveit.core.robot_state import RobotState
from moveit.core.planning_scene import PlanningScene

from std_msgs.msg import Float64MultiArray, Bool
from std_srvs.srv import SetBool
from easy_interfaces.srv import SolveIK, CheckCollision
from controller_manager_msgs.srv import SwitchController

import copy

IK_FAILURE_PENALTY = 1.0
COLLISION_PENALTY = 1.0
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
        
    def set_action(self, action):
        super().set_action(action)
        # Enable or disable expert based on mode
        enable_expert_req = SetBool.Request()
        enable_expert_req.data = (self._mode == AgentMode.BEHAVIOR_CLONING)
        
        if self.action == "pick":
            self.enable_picking_expert_client.call_async(enable_expert_req)
        elif self.action == "place":
            self.enable_placing_expert_client.call_async(enable_expert_req)
        
        
    def set_mode(self, mode: AgentMode):
        self._mode = mode
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
    def perform(self, act: dict, state_interface: StateInterface) -> tuple[float, dict]:
        state = copy.deepcopy(state_interface.get_state())  # Get current state
        reward = 0.0
        if act:
            print(f"[SkillExecutionActionInterface] Performing action: {act}", flush=True)
            
            EEF_diff = transform_distance(
                state["eef_pose"],
                act["eef_pose"]
            )
            reward -= EEF_diff * EEF_MOVEMENT_PENALTY_SCALE  # Penalize large movements of the end-effector
            
            joint_positions = self.solve_IK(act["eef_pose"], state["joint_positions"])
            if not joint_positions:
                reward -= IK_FAILURE_PENALTY  # Penalize for IK failure
                
            if joint_positions:
                # Feasible for execution
                taken_act = act
                
                self.send_joint_command(joint_positions)
                self.send_gripper_command(act["suction_command"])
            else:
                # IK failure, wait for expert demonstration or next state
                print(f"[SkillExecutionActionInterface] IK failure, wait for expert demonstration or next state", flush=True)
                taken_act = {
                    "eef_pose": state["eef_pose"],
                    "suction_command": state["cmd_suction_state"]
                }
            
            self.wait_for_next_state()
            if joint_positions and self.check_collision(joint_positions):
                reward -= COLLISION_PENALTY  # Penalize for collision
        
        if not act:
            print(f"[SkillExecutionActionInterface] No action provided, wait for expert demonstration or next state", flush=True)
            while self._mode == AgentMode.BEHAVIOR_CLONING:
                self.wait_for_next_state()
                next_state = state_interface.get_state()  # Get current state
                taken_act = {
                    "eef_pose": next_state["eef_pose"],
                    "suction_command": next_state["cmd_suction_state"]
                }
                
                # Dont record expert demonstration if no movement is taken
                tf_dist = transform_distance(state["eef_pose"], next_state["eef_pose"])
                print(f"[SkillExecutionActionInterface] Waiting for expert demonstration... Current EEF distance moved: {tf_dist:.4f}, suction command: {taken_act['suction_command']}", flush=True)
                print(f"[SkillExecutionActionInterface] Current state: {state['eef_pose']}", flush=True)
                print(f"[SkillExecutionActionInterface] Next state: {next_state['eef_pose']}", flush=True)
                if tf_dist < 1e-3 and \
                        taken_act["suction_command"]  == state["cmd_suction_state"]:
                    continue
                
                break
            
            if self._mode != AgentMode.BEHAVIOR_CLONING:
                return 0.0, {}  # No action taken, no reward when switching mode
            
        # Accumulate reward from next state
        reward += state_interface.get_reward()
        
        return reward, taken_act
    
    
    def solve_IK(self, target_pose: list[float], initial_joint_positions: list[float]) -> list[float]:
        print(f"[SkillExecutionActionInterface] Solving IK for target pose: {target_pose}", flush=True)
        req = SolveIK.Request()
        req.group_name = MOVEGROUP_NAME
        req.target_pose = target_pose
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
        print(f"[SkillExecutionActionInterface] Checking collision for joint positions: {joint_positions}", flush=True)
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
        print(f"[SkillExecutionActionInterface] Sending joint command: {joint_positions}", flush=True)
        joint_command_msg = Float64MultiArray()
        joint_command_msg.data = joint_positions
        self.joint_command_pub.publish(joint_command_msg)
        
        
    def send_gripper_command(self, suction_command: float):
        suction_on = suction_command > 0.5
        print(f"[SkillExecutionActionInterface] Sending gripper command: {'Suction ON' if suction_on else 'Suction OFF'}, value: {suction_command}", flush=True)
        suction_command_msg = Bool()
        suction_command_msg.data = suction_on
        self.cmd_suction_pub.publish(suction_command_msg)
