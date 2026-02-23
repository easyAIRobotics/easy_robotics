#include <rclcpp/rclcpp.hpp>
#include "rclcpp/executors/multi_threaded_executor.hpp"

#include <moveit/move_group_interface/move_group_interface.hpp>
#include <moveit/planning_scene_interface/planning_scene_interface.hpp>
#include <moveit/planning_scene/planning_scene.hpp>
#include <moveit/robot_model_loader/robot_model_loader.hpp>
#include <moveit/robot_state/robot_state.hpp>
#include <moveit/kinematics_base/kinematics_base.hpp>
#include <moveit/collision_detection/collision_common.hpp>

#include <tf2_eigen/tf2_eigen.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

#include "easy_interfaces/srv/solve_ik.hpp"
#include "easy_interfaces/srv/check_collision.hpp"

using SolveIK = easy_interfaces::srv::SolveIK;
using CheckCollision = easy_interfaces::srv::CheckCollision;

class MoveItExecutorNode
{
public:
  MoveItExecutorNode(rclcpp::Node::SharedPtr &node)
      : node_(node)
  {
    // -------------------------------
    // Planning scene monitor
    // -------------------------------
    move_group_ = std::make_shared<moveit::planning_interface::MoveGroupInterface>(node_, planning_group_);
    planning_scene_interface_ = std::make_shared<moveit::planning_interface::PlanningSceneInterface>();
    // Load robot model
    robot_model_loader::RobotModelLoader robot_model_loader(node_->shared_from_this(), "robot_description");
    kinematics_model = robot_model_loader.getModel();
    kinematics_state = std::make_shared<moveit::core::RobotState>(kinematics_model);
    planning_scene_ = std::make_shared<planning_scene::PlanningScene>(kinematics_model);
    if (!move_group_)
    {
      RCLCPP_ERROR(node_->get_logger(), "Failed to create MoveGroupInterface for planning group: %s", planning_group_.c_str());
      throw std::runtime_error("MoveGroupInterface initialization failed");
    }
    move_group_->startStateMonitor();
    move_group_->setPoseReferenceFrame("base_link");

    // -------------------------------
    // Services
    // -------------------------------
    service_cb_group_ = node->create_callback_group(
        rclcpp::CallbackGroupType::MutuallyExclusive);

    solve_ik_srv_ = node_->create_service<SolveIK>(
        "solve_ik",
        std::bind(&MoveItExecutorNode::handleSolveIK,
                  this,
                  std::placeholders::_1,
                  std::placeholders::_2),
        rclcpp::ServicesQoS(),
        service_cb_group_);

    tf_buffer_ = std::make_shared<tf2_ros::Buffer>(node_->get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

    check_collision_srv_ = node_->create_service<CheckCollision>(
        "check_collision",
        std::bind(&MoveItExecutorNode::handleCheckCollision,
                  this,
                  std::placeholders::_1,
                  std::placeholders::_2),
        rclcpp::ServicesQoS(),
        service_cb_group_);

    RCLCPP_INFO(node_->get_logger(), "SolveIK + CheckCollision services ready");
  }

private:
  std::shared_ptr<rclcpp::Node> node_;
  std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
  // ==========================================================
  // Solve IK Service
  // ==========================================================
  void handleSolveIK(
      const std::shared_ptr<SolveIK::Request> req,
      std::shared_ptr<SolveIK::Response> res)
  {
    RCLCPP_INFO(node_->get_logger(), "Received SolveIK request for group '%s'", req->group_name.c_str());
    // -------------------------------
    // Build RobotState with seed
    // -------------------------------
    const moveit::core::JointModelGroup *jmg =
        kinematics_model->getJointModelGroup(planning_group_);
    kinematics_state->setJointGroupPositions(jmg, req->initial_joint_positions);
    kinematics_state->update();
    RCLCPP_INFO(node_->get_logger(), "Kinematics state updated");

    if (!jmg)
    {
      res->success = false;
      res->message = "Invalid planning group";
      return;
    }

    if (req->initial_joint_positions.size() != jmg->getVariableCount())
    {
      res->success = false;
      res->message = "Initial joint position size mismatch";
      return;
    }

    std::vector<double> seed_joint_values;
    kinematics_state->copyJointGroupPositions(jmg, seed_joint_values);
    RCLCPP_INFO(node_->get_logger(), "Seed joint values:");
    for (size_t i = 0; i < seed_joint_values.size(); ++i)
    {
      RCLCPP_INFO(node_->get_logger(), "  Joint %zu: %f", i + 1, seed_joint_values[i]);
    }

    // -------------------------------
    // Target pose
    // -------------------------------
    geometry_msgs::msg::Pose target_pose;
    target_pose.position.x = req->target_pose[0];
    target_pose.position.y = req->target_pose[1];
    target_pose.position.z = req->target_pose[2];
    target_pose.orientation.x = req->target_pose[3];
    target_pose.orientation.y = req->target_pose[4];
    target_pose.orientation.z = req->target_pose[5];
    target_pose.orientation.w = req->target_pose[6];
    RCLCPP_INFO(node_->get_logger(), "Target pose: position(%.3f, %.3f, %.3f), orientation(%.3f, %.3f, %.3f, %.3f)",
                target_pose.position.x, target_pose.position.y, target_pose.position.z,
                target_pose.orientation.x, target_pose.orientation.y, target_pose.orientation.z, target_pose.orientation.w);

    // -------------------------------
    // Solve IK (seeded)
    // -------------------------------
    bool found_ik = false;
    for (size_t i = 0; i < 1; ++i)
    {
      found_ik = kinematics_state->setFromIK(
          jmg, transformPoseToWorld(target_pose),
          0.2);

      if (found_ik)
        break;
    }

    if (!found_ik)
    {
      res->success = false;
      res->message = "IK failed";
      return;
    }

    // -------------------------------
    // Collision check
    // -------------------------------
    collision_detection::CollisionRequest creq;
    collision_detection::CollisionResult cres;

    planning_scene_->checkCollision(creq, cres, *kinematics_state);

    res->success = true;
    res->in_collision = cres.collision;
    res->message = cres.collision ? "IK solution found but in collision" : "IK solution valid and collision-free";

    // -------------------------------
    // Output joints
    // -------------------------------
    res->joint_positions.resize(jmg->getVariableCount());
    kinematics_state->copyJointGroupPositions(jmg, res->joint_positions);
  }

  // ==========================================================
  // Check Collision Service
  // ==========================================================
  void handleCheckCollision(
      const std::shared_ptr<CheckCollision::Request> req,
      std::shared_ptr<CheckCollision::Response> res)
  {
    const moveit::core::JointModelGroup *jmg =
        kinematics_model->getJointModelGroup(planning_group_);

    if (!jmg ||
        req->joint_positions.size() != jmg->getVariableCount())
    {
      res->in_collision = true;
      return;
    }

    moveit::core::RobotState state(kinematics_model);
    state.setJointGroupPositions(jmg, req->joint_positions);
    state.update();

    collision_detection::CollisionRequest creq;
    collision_detection::CollisionResult cres;

    planning_scene_->checkCollision(creq, cres, state);

    res->in_collision = cres.collision;
  }

  geometry_msgs::msg::Pose transformPoseToWorld(
      const geometry_msgs::msg::Pose &pose_in_base)
  {
    geometry_msgs::msg::PoseStamped pose_base;
    pose_base.header.frame_id = "base_link";
    pose_base.header.stamp = node_->get_clock()->now();
    pose_base.pose = pose_in_base;

    geometry_msgs::msg::PoseStamped pose_world;

    try
    {
      pose_world = tf_buffer_->transform(
          pose_base,
          "world",
          tf2::durationFromSec(0.2));
    }
    catch (tf2::TransformException &ex)
    {
      RCLCPP_ERROR(node_->get_logger(),
                   "TF transform failed (base_link → world): %s",
                   ex.what());
      throw;
    }

    return pose_world.pose;
  }

  // ==========================================================
  std::shared_ptr<moveit::planning_interface::MoveGroupInterface> move_group_;
  std::shared_ptr<moveit::planning_interface::PlanningSceneInterface> planning_scene_interface_;
  planning_scene::PlanningScenePtr planning_scene_;
  moveit::core::RobotModelPtr kinematics_model;
  moveit::core::RobotStatePtr kinematics_state;
  std::string planning_group_ = "suction_tip";

  rclcpp::Service<SolveIK>::SharedPtr solve_ik_srv_;
  rclcpp::Service<CheckCollision>::SharedPtr check_collision_srv_;
  rclcpp::CallbackGroup::SharedPtr service_cb_group_;
};

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<rclcpp::Node>("moveit_executor_node");
  auto executor_node = std::make_shared<MoveItExecutorNode>(node);
  rclcpp::executors::MultiThreadedExecutor m_executor; // allows multiple callbacks in parallel
  m_executor.add_node(node);
  m_executor.spin();
  rclcpp::shutdown();
  return 0;
}
