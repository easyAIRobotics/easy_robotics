#pragma once

#include "easy_behaviors/headers.hpp"
#include <random>
#include <fstream>

namespace easy_behaviors
{
    class ObjectSelectorNode : public BT::SyncActionNode
    {
    public:
        ObjectSelectorNode(const std::string& name, const BT::NodeConfiguration& config);

        static BT::PortsList providedPorts()
        {
            return {
                BT::InputPort<std::string>("input_type", "The type of target (e.g., 'box' or 'container')"),
                BT::OutputPort<std::string>("output_type", "The type of next target (e.g., 'brown_box' -> 'brown container', etc.)"),
                BT::OutputPort<easy_interfaces::msg::BoundingBox>("selected_bbox")
            };
        }

        virtual BT::NodeStatus tick() override;
    private:
        rclcpp::Node::SharedPtr node_;
        easy_interfaces::msg::BoundingBoxes latest_bboxes_;
        rclcpp::Publisher<easy_interfaces::msg::BoundingBox>::SharedPtr action_object_publisher_;
        rclcpp::Subscription<easy_interfaces::msg::BoundingBoxes>::SharedPtr bboxes_subscription_;

        const std::unordered_map<std::string, std::string> type_mapping_ = {
            {"brown_box", "black_container"},
            {"green_box", "green_container"},
            {"black_container", "box"},
            {"green_container", "box"}
        };
    };

    class PolicyNode : public BT::StatefulActionNode
    {
    public:
        PolicyNode(const std::string& name, const BT::NodeConfiguration& config);

        static BT::PortsList providedPorts()
        {
            return {
                BT::InputPort<easy_interfaces::msg::BoundingBox>("bbox", "The bounding box of the selected object"),
                BT::InputPort<std::string>("skill", "pick or place"),
                BT::InputPort<std::string>("mode", "Policy or expert mode (e.g., 'policy', 'expert', or 'policy_d')"),
            };
        }

        virtual BT::NodeStatus onStart() override;
        virtual BT::NodeStatus onRunning() override;
        virtual void onHalted() override;

    private:
        rclcpp::Node::SharedPtr node_;
        std::unordered_map<std::string, std::string> mode_mapping_ = {{"policy", "training/auto"}, {"expert", "training/manual"}, {"policy_d", "training/exec"}};  // Mapping from input mode to skill execution mode
        bool action_done_ = false, done_state_reset_ = false;  // Flag to indicate if action is done
        std::string _mode, _skill;
        std::chrono::time_point<std::chrono::steady_clock> action_start_time_;

        rclcpp::Publisher<easy_interfaces::msg::Pixel>::SharedPtr action_point_publisher_;
        rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr action_result_subscription_;
        
        rclcpp::Client<easy_interfaces::srv::SetString>::SharedPtr set_skill_client_;
        rclcpp::Client<easy_interfaces::srv::SetString>::SharedPtr set_mode_client_;
    };

    class HomingNode : public BT::StatefulActionNode
    {
    public:
        HomingNode(const std::string& name, const BT::NodeConfiguration& config);
        static BT::PortsList providedPorts()
        {
            return {
                BT::InputPort<std::string>("where", "home or random"),
            };
        }
        virtual BT::NodeStatus onStart() override;
        virtual BT::NodeStatus onRunning() override;
        virtual void onHalted() override;
    private:
        rclcpp::Node::SharedPtr node_;
        bool done_ = false;
        easy_interfaces::srv::ExecuteJointGoal::Response::SharedPtr result_;
        rclcpp::Client<easy_interfaces::srv::ExecuteJointGoal>::SharedPtr execute_joint_goal_client_;
        rclcpp::Client<easy_interfaces::srv::ExecuteRandomGoal>::SharedPtr execute_random_goal_client_;
    };

    class ResetNode : public BT::SyncActionNode
    {
    public:
        ResetNode(const std::string& name, const BT::NodeConfiguration& config);
        static BT::PortsList providedPorts()
        {
            return {};
        }
        virtual BT::NodeStatus tick() override;
    private:
        rclcpp::Node::SharedPtr node_;

        rclcpp::Client<std_srvs::srv::Trigger>::SharedPtr reset_client_;
    };

    class TriggerGripperNode : public BT::SyncActionNode
    {
    public:
        TriggerGripperNode(const std::string& name, const BT::NodeConfiguration& config);
        static BT::PortsList providedPorts()
        {
            return {
                BT::InputPort<bool>("active", "True to active the gripper, False to deactivate the gripper")
            };
        }
        virtual BT::NodeStatus tick() override;
    private:
        rclcpp::Node::SharedPtr node_;
        rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr gripper_command_publisher_;
    };

    class SetModeNode : public BT::SyncActionNode
    {
    public:
        SetModeNode(const std::string& name, const BT::NodeConfiguration& config);
        static BT::PortsList providedPorts()
        {
            return {
                BT::InputPort<std::string>("mode", "The mode to set (e.g., 'training/auto' or 'training/manual')")
            };
        }
        virtual BT::NodeStatus tick() override;
    private:
        rclcpp::Node::SharedPtr node_;
        rclcpp::Client<easy_interfaces::srv::SetString>::SharedPtr set_mode_client_;
    };

    class CounterNode : public BT::SyncActionNode
    {
    public:
        CounterNode(const std::string& name, const BT::NodeConfiguration& config);
        static BT::PortsList providedPorts()
        {
            return {
                BT::InputPort<std::string>("key", "The key to identify the counter"),
                BT::InputPort<int>("increment", "The value to increment the counter (default: 1)"),
                BT::OutputPort<int>("count", "The current count value after increment")
            };
        }
        virtual BT::NodeStatus tick() override;

    private:
        rclcpp::Node::SharedPtr node_;
        std::unordered_map<std::string, int> counters_;  // Map to store counters
    };

    class TimeCounterNode : public BT::SyncActionNode
    {
    public:
        TimeCounterNode(const std::string& name, const BT::NodeConfiguration& config);
        static BT::PortsList providedPorts()
        {
            return {
                BT::InputPort<std::string>("key", "The key to identify the timer"),
                BT::InputPort<std::string>("switch", "The duration of the timer in seconds"),
                BT::InputPort<double>("input_time", "The start time recorded for the key"),
                BT::OutputPort<double>("output_time", "The start time recorded for the key")
            };
        }
        virtual BT::NodeStatus tick() override;

    private:
        rclcpp::Node::SharedPtr node_;
        std::unordered_map<std::string, std::vector<double>> durations;  // Store execution durations
        std::unordered_map<std::string, double> start_times;  // Store start times

        void printStatistics(const std::string& key);
        void appendCSV(const std::string& key, double elapsed_time);
    };
}  // namespace easy_behaviors
