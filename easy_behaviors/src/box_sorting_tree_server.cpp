#include <chrono>
#include <fstream>
#include <random>
#include <string>
#include <memory>

#include <behaviortree_cpp/bt_factory.h>
#include <behaviortree_cpp/loggers/bt_cout_logger.h>
#include <behaviortree_cpp/loggers/groot2_publisher.h>
#include <behaviortree_cpp/xml_parsing.h>

#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>
#include "easy_behaviors/policy_bt_nodes.hpp"

using namespace easy_behaviors;

class BoxSortingTreeServer : public rclcpp::Node
{
private:
    std::shared_ptr<BT::BehaviorTreeFactory> factory_;
    BT::Tree tree_;

public:
    BoxSortingTreeServer() : Node("box_sorting_tree_server")
    {
        // Initialize the Behavior Tree factory and register custom nodes
        factory_ = std::make_shared<BT::BehaviorTreeFactory>();
        factory_->registerNodeType<easy_behaviors::ObjectSelectorNode>("ObjectSelector");
        factory_->registerNodeType<easy_behaviors::PolicyNode>("Policy");
        factory_->registerNodeType<easy_behaviors::HomingNode>("Homing");
        factory_->registerNodeType<easy_behaviors::ResetNode>("Reset");
        factory_->registerNodeType<easy_behaviors::TriggerGripperNode>("TriggerGripper");
        factory_->registerNodeType<easy_behaviors::SetModeNode>("SetMode");
        factory_->registerNodeType<easy_behaviors::CounterNode>("Counter");
        factory_->registerNodeType<easy_behaviors::TimeCounterNode>("TimeCounter");
        // Load the Behavior Tree from an XML file
        std::string xml_path;
        this->declare_parameter<std::string>("tree_path", "");
        this->get_parameter("tree_path", xml_path);

        tree_ = factory_->createTreeFromFile(xml_path);
    }

    void executeTree()
    {
        std::this_thread::sleep_for(std::chrono::seconds(1)); // Give some time for the node to initialize
        BT::NodeStatus status = BT::NodeStatus::RUNNING;
        while (rclcpp::ok()) {
            status = tree_.rootNode()->executeTick();
        }
        RCLCPP_INFO(this->get_logger(), "Behavior Tree execution finished with status: %s", BT::toStr(status).c_str());
    }
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto tree = std::make_shared<BoxSortingTreeServer>();
    tree->executeTree();
    rclcpp::shutdown();
    return 0;
}
