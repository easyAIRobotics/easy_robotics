#include "easy_behaviors/policy_bt_nodes.hpp"

namespace easy_behaviors
{
    ObjectSelectorNode::ObjectSelectorNode(const std::string &name, const BT::NodeConfiguration &config)
        : node_(std::make_shared<rclcpp::Node>("object_selector_node")),
          BT::SyncActionNode(name, config)
    {
        action_object_publisher_ = node_->create_publisher<easy_interfaces::msg::BoundingBox>("selected_box", 1);

        // Subscribe to the bounding boxes topic
        bboxes_subscription_ = node_->create_subscription<easy_interfaces::msg::BoundingBoxes>(
            "easy_object_detection/bounding_boxes", 1,
            [this](const easy_interfaces::msg::BoundingBoxes::SharedPtr msg)
            {
                // Store the latest bounding boxes message
                latest_bboxes_ = *msg;
            });
    }

    BT::NodeStatus ObjectSelectorNode::tick()
    {
        rclcpp::spin_some(node_); // Process incoming messages to update bounding boxes
        // Get the input type
        std::string target_type;
        if (!getInput<std::string>("input_type", target_type))
        {
            return BT::NodeStatus::FAILURE;
        }

        if (latest_bboxes_.bboxes.empty())
        {
            return BT::NodeStatus::FAILURE;
        }
        RCLCPP_INFO(node_->get_logger(), "ObjectSelectorNode: Received %zu bounding boxes, selecting type '%s'", latest_bboxes_.bboxes.size(), target_type.c_str());
        // Filter list of bounding boxes based on the input type
        std::vector<easy_interfaces::msg::BoundingBox> filtered_bboxes;
        for (const auto &bbox : latest_bboxes_.bboxes)
        {
            // Case type == box, accept any class with "box" in its name
            if (target_type == "box" && bbox.class_id.find("box") != std::string::npos)
            {
                if (bbox.x < 240 || bbox.x > 460) continue;
                int dx = static_cast<int>(bbox.x) - 320;
                int dy = static_cast<int>(bbox.y) - 420;
                double distance = std::sqrt(dx * dx + dy * dy);
                if (distance < 60 || distance > 200) continue;
                if (bbox.y > 400) continue;

                filtered_bboxes.push_back(bbox);
            }
            // Case type == container, only accept the corresponding class
            else if (target_type == bbox.class_id)
            {
                if (bbox.x < 40 || bbox.x > 600) continue;
                filtered_bboxes.push_back(bbox);
            }
        }
        latest_bboxes_.bboxes.clear(); // Clear the stored bounding boxes after processing

        // No bounding boxes found for the specified type
        if (filtered_bboxes.empty())
        {
            RCLCPP_WARN(node_->get_logger(), "ObjectSelectorNode: No bounding box found for type '%s'", target_type.c_str());
            return BT::NodeStatus::FAILURE;
        }

        // Randomly select one bounding box from the filtered list
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(0, filtered_bboxes.size() - 1);
        const auto &selected_bbox = filtered_bboxes[dis(gen)];

        // Set the output ports
        setOutput("selected_bbox", selected_bbox);
        auto it = type_mapping_.find(selected_bbox.class_id);
        if (it != type_mapping_.end())
        {
            setOutput("output_type", it->second);
        }
        else
        {
            setOutput("output_type", selected_bbox.class_id);
        }
        // Publish the selected bounding box for the action action
        action_object_publisher_->publish(selected_bbox);
        RCLCPP_INFO(node_->get_logger(), "ObjectSelectorNode: Published bounding box with class '%s' at (x: %.2f, y: %.2f)", 
            selected_bbox.class_id.c_str(), selected_bbox.x, selected_bbox.y);
        RCLCPP_INFO(node_->get_logger(), "ObjectSelectorNode: Output type set to '%s'", 
            it != type_mapping_.end() ? it->second.c_str() : selected_bbox.class_id.c_str());
        return BT::NodeStatus::SUCCESS;
    }

    PolicyNode::PolicyNode(const std::string &name, const BT::NodeConfiguration &config)
        : node_(std::make_shared<rclcpp::Node>("policy_node")),
          BT::StatefulActionNode(name, config)
    {
        action_point_publisher_ = node_->create_publisher<easy_interfaces::msg::Pixel>("selected_point", 1);
        action_result_subscription_ = node_->create_subscription<std_msgs::msg::Bool>(
            "skill_execution/result", 1,
            [this](const std_msgs::msg::Bool::SharedPtr msg)
            {
                action_done_ = msg->data;
            });
        set_skill_client_ = node_->create_client<easy_interfaces::srv::SetString>("skill_execution/set_action");
        set_mode_client_ = node_->create_client<easy_interfaces::srv::SetString>("skill_execution/set_mode");
    }

    BT::NodeStatus PolicyNode::onStart()
    {
        // Get the input bounding box
        easy_interfaces::msg::BoundingBox bbox;
        if (!getInput<easy_interfaces::msg::BoundingBox>("bbox", bbox))
        {
            RCLCPP_ERROR(node_->get_logger(), "PolicyNode: Missing input 'bbox'");
            return BT::NodeStatus::FAILURE;
        }
        if (!getInput<std::string>("mode", _mode))
        {
            RCLCPP_ERROR(node_->get_logger(), "PolicyNode: Missing input 'mode'");
            return BT::NodeStatus::FAILURE;
        }
        if (!getInput<std::string>("skill", _skill))
        {
            RCLCPP_ERROR(node_->get_logger(), "PolicyNode: Missing input 'skill'");
            return BT::NodeStatus::FAILURE;
        }
        
        // Set the skill and mode for action (this is just an example, adjust as needed)
        set_skill_client_->wait_for_service();
        auto skill_request = std::make_shared<easy_interfaces::srv::SetString::Request>();
        skill_request->data = _skill;
        set_skill_client_->async_send_request(skill_request);

        set_mode_client_->wait_for_service();
        auto mode_request = std::make_shared<easy_interfaces::srv::SetString::Request>();

        mode_request->data = mode_mapping_[_mode];
        set_mode_client_->async_send_request(mode_request);

        if (_mode == "expert")
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(1000)); // Small delay to ensure the mode is set before publishing the action point
            // Publish action point based on the bounding box center
            easy_interfaces::msg::Pixel action_point;
            action_point.px = static_cast<int>(bbox.x);
            action_point.py = static_cast<int>(bbox.y);
            action_point_publisher_->publish(action_point);
            std::this_thread::sleep_for(std::chrono::milliseconds(1000));
            action_point_publisher_->publish(action_point); // Publish again to ensure the message is received
        }

        // Wait for done or timeout
        done_state_reset_ = false; // Reset the done state reset flag
        action_start_time_ = std::chrono::steady_clock::now(); // Record the start time of the action
        return BT::NodeStatus::RUNNING;
    }

    BT::NodeStatus PolicyNode::onRunning()
    {
        rclcpp::spin_some(node_);
        if (!done_state_reset_ && !action_done_)
        {
            RCLCPP_INFO(node_->get_logger(), "PolicyNode (skill - %s, mode - %s): Done state reset", _skill.c_str(), _mode.c_str());
            done_state_reset_ = true;
            action_done_ = false;
            return BT::NodeStatus::RUNNING;
        }

        auto now = std::chrono::steady_clock::now();
        double elapsed_seconds = std::chrono::duration_cast<std::chrono::duration<double>>(now - action_start_time_).count();
        if (elapsed_seconds > 5.0 && !done_state_reset_)
        {
            done_state_reset_ = true; // Set the flag to indicate that we've reset the done state after timeout
            action_done_ = false;
            return BT::NodeStatus::RUNNING;
        }
        
        if (done_state_reset_ && action_done_)
        {
            RCLCPP_INFO(node_->get_logger(), "PolicyNode (skill - %s, mode - %s): Action execution completed successfully", _skill.c_str(), _mode.c_str());
            // auto mode_request = std::make_shared<easy_interfaces::srv::SetString::Request>();
            // mode_request->data = "training/stop";
            // set_mode_client_->async_send_request(mode_request);
            return BT::NodeStatus::SUCCESS;
        }

        return BT::NodeStatus::RUNNING;
    }

    void PolicyNode::onHalted()
    {
        RCLCPP_INFO(node_->get_logger(), "PolicyNode (skill - %s, mode - %s): Halted, stopping skill execution", _skill.c_str(), _mode.c_str());
        auto mode_request = std::make_shared<easy_interfaces::srv::SetString::Request>();

        mode_request->data = "training/stop";
        set_mode_client_->async_send_request(mode_request);
    }

    HomingNode::HomingNode(const std::string &name, const BT::NodeConfiguration &config)
        : node_(std::make_shared<rclcpp::Node>("homing_node")),
          BT::StatefulActionNode(name, config)
    {
        execute_joint_goal_client_ = node_->create_client<easy_interfaces::srv::ExecuteJointGoal>("execute_joint_goal");
        execute_random_goal_client_ = node_->create_client<easy_interfaces::srv::ExecuteRandomGoal>("execute_random_goal");
    
    }

    BT::NodeStatus HomingNode::onStart()
    {
        std::string where;
        if (!getInput<std::string>("where", where))
        {
            RCLCPP_ERROR(node_->get_logger(), "HomingNode: Missing input 'where'");
            return BT::NodeStatus::FAILURE;
        }

        if (where == "home")
        {
            RCLCPP_INFO(node_->get_logger(), "HomingNode: Starting homing action");
            // Define the home joint positions (this is just an example, adjust as needed)
            std::vector<double> home_joint_positions = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

            // Create and send the service request to execute the joint goal
            auto request = std::make_shared<easy_interfaces::srv::ExecuteJointGoal::Request>();
            request->joint_goal = home_joint_positions;
            request->planning_time = 5.0; // Set a reasonable planning time
            request->speed_factor = 0.4;  // Set speed factor to normal

            execute_joint_goal_client_->async_send_request(
                request,
                [this](rclcpp::Client<easy_interfaces::srv::ExecuteJointGoal>::SharedFuture future)
                {
                    auto response = future.get();
                    result_ = response;
                    done_ = true;
                });
        }
        else if (where == "random")
        {
            RCLCPP_INFO(node_->get_logger(), "HomingNode: Starting random pose action");
            auto request = std::make_shared<easy_interfaces::srv::ExecuteRandomGoal::Request>();
            request->planning_time = 3.0; // Set a reasonable planning time
            request->speed_factor = 0.4;  // Set speed factor to normal

            execute_random_goal_client_->async_send_request(
                request,
                [this](rclcpp::Client<easy_interfaces::srv::ExecuteRandomGoal>::SharedFuture future)
                {
                    auto response = future.get();
                    result_ = std::make_shared<easy_interfaces::srv::ExecuteJointGoal::Response>();
                    result_->success = response->success;
                    result_->message = response->message;
                    done_ = true;
                });
            
            done_ = false;
            RCLCPP_INFO(node_->get_logger(), "HomingNode: Sent random pose goal, waiting for result...");
        }
        else
        {
            RCLCPP_ERROR(node_->get_logger(), "HomingNode: Invalid input for 'where': %s", where.c_str());
            return BT::NodeStatus::FAILURE;
        }

        done_ = false;
        return BT::NodeStatus::RUNNING;
    }

    BT::NodeStatus HomingNode::onRunning()
    {
        rclcpp::spin_some(node_);
        if (!done_)
            return BT::NodeStatus::RUNNING;

        if (!result_->success)
            return BT::NodeStatus::FAILURE;

        RCLCPP_INFO(node_->get_logger(), "HomingNode: Homing action completed successfully");

        return BT::NodeStatus::SUCCESS;
    }

    void HomingNode::onHalted()
    {
        RCLCPP_INFO(node_->get_logger(), "HomingNode: Halted");
    }

    ResetNode::ResetNode(const std::string &name, const BT::NodeConfiguration &config)
        : node_(std::make_shared<rclcpp::Node>("reset_node")),
          BT::SyncActionNode(name, config)
    {
        reset_client_ = node_->create_client<std_srvs::srv::Trigger>("respawn_boxes");
    }

    BT::NodeStatus ResetNode::tick()
    {
        auto request = std::make_shared<std_srvs::srv::Trigger::Request>();

        auto future = reset_client_->async_send_request(request);
        if (rclcpp::spin_until_future_complete(node_, future) != rclcpp::FutureReturnCode::SUCCESS)
        {
            RCLCPP_ERROR(node_->get_logger(), "ResetNode: Failed to call service 'respawn_boxes'");
            return BT::NodeStatus::FAILURE;
        }

        auto response = future.get();
        if (!response->success)
        {
            RCLCPP_ERROR(node_->get_logger(), "ResetNode: Service 'respawn_boxes' failed with message: %s", response->message.c_str());
            return BT::NodeStatus::FAILURE;
        }

        return BT::NodeStatus::SUCCESS;
    }

    TriggerGripperNode::TriggerGripperNode(const std::string &name, const BT::NodeConfiguration &config)
        : node_(std::make_shared<rclcpp::Node>("trigger_gripper_node")),
          BT::SyncActionNode(name, config)
    {
        gripper_command_publisher_ = node_->create_publisher<std_msgs::msg::Bool>("cmd_suction", 1);
    }

    BT::NodeStatus TriggerGripperNode::tick()
    {
        // Get the input command
        bool active;
        if (!getInput<bool>("active", active))
        {
            RCLCPP_ERROR(node_->get_logger(), "TriggerGripperNode: Missing input 'active'");
            return BT::NodeStatus::FAILURE;
        }

        // Publish the gripper command
        std_msgs::msg::Bool command_msg;
        command_msg.data = active;
        gripper_command_publisher_->publish(command_msg);
        RCLCPP_INFO(node_->get_logger(), "TriggerGripperNode: Published gripper command to %s the gripper", active ? "active" : "deactive");
        return BT::NodeStatus::SUCCESS;
    }

    SetModeNode::SetModeNode(const std::string &name, const BT::NodeConfiguration &config)
        : node_(std::make_shared<rclcpp::Node>("set_mode_node")),
          BT::SyncActionNode(name, config)
    {
        set_mode_client_ = node_->create_client<easy_interfaces::srv::SetString>("skill_execution/set_mode");
    }

    BT::NodeStatus SetModeNode::tick()
    {
        // Get the input mode
        std::string mode;
        if (!getInput<std::string>("mode", mode))
        {
            RCLCPP_ERROR(node_->get_logger(), "SetModeNode: Missing input 'mode'");
            return BT::NodeStatus::FAILURE;
        }

        // Send the mode setting request
        auto request = std::make_shared<easy_interfaces::srv::SetString::Request>();
        request->data = mode;
        set_mode_client_->async_send_request(request);
        RCLCPP_INFO(node_->get_logger(), "SetModeNode: Sent request to set mode to '%s'", mode.c_str());
        return BT::NodeStatus::SUCCESS;
    }

    CounterNode::CounterNode(const std::string &name, const BT::NodeConfiguration &config)
        : node_(std::make_shared<rclcpp::Node>("counter_node")),
          BT::SyncActionNode(name, config)
    {
    }

    BT::NodeStatus CounterNode::tick()
    {
        // Get the input key and increment value
        std::string key;
        int increment = 1; // Default increment value
        if (!getInput<std::string>("key", key))
        {
            RCLCPP_ERROR(node_->get_logger(), "CounterNode: Missing input 'key'");
            return BT::NodeStatus::FAILURE;
        }
        getInput<int>("increment", increment); // Optional input, will use default if not provided

        // Increment the counter for the given key
        int count = counters_[key] += increment;

        // Set the output count value
        setOutput("count", count);
        RCLCPP_INFO(node_->get_logger(), "CounterNode: Key '%s' incremented by %d, current count is %d", key.c_str(), increment, count);
        return BT::NodeStatus::SUCCESS;
    }
} // namespace easy_behaviors
