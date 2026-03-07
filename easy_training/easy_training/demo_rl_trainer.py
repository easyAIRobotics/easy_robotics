import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup

from easy_interfaces.srv import SetString

from easy_training.skill_execution.skill_execution_agent_interface import SkillExecutionAgentInterface

import threading

AGENT_REGISTRY = {
    "SkillExecution": SkillExecutionAgentInterface,
    # "AnotherAgent": AnotherAgentInterface,
}

class DemoRLTrainer(Node):
    def __init__(self):
        super().__init__("demo_rl_trainer")
        
        self.set_mode_callback_group = ReentrantCallbackGroup()
        
        self.declare_parameter("agent_name", "skill_execution")
        self.declare_parameter("agent_type", "SkillExecution")
        
        self.agent_name = self.get_parameter("agent_name").get_parameter_value().string_value
        self.agent_type = self.get_parameter("agent_type").get_parameter_value().string_value
        
        agent_cls = AGENT_REGISTRY.get(self.agent_type)

        if agent_cls is None:
            self.get_logger().error(f"Unsupported agent type: {self.agent_type}")
            rclpy.shutdown()
            return

        self.agent = agent_cls(self, self.agent_name)
        
        # ROS2 services for controlling the agent mode
        self.create_service(
            srv_type=SetString,
            srv_name=f"{self.agent_name}/set_mode",
            callback=self.set_agent_mode_callback,
            callback_group=self.set_mode_callback_group
        )
        
        self.create_service(
            srv_type=SetString,
            srv_name=f"{self.agent_name}/command",
            callback=self.set_agent_command_callback,
            callback_group=self.set_mode_callback_group
        )
        
        self.get_logger().info(f"Demo RL Trainer node for agent {self.agent_name} has been started.")
        
    
    def set_agent_mode_callback(self, request, response):
        self.get_logger().info(f"Received request to set agent mode to: {request.data}")
        self.agent.set_mode(request.data)
        response.success = True
        response.message = f"Agent {self.agent_name} mode set to {request.data}"
        return response
    
    
    def set_agent_command_callback(self, request, response):
        self.get_logger().info(f"Received command for agent: {request.data}")
        # Here you can implement handling of specific commands, e.g., start training, save model, etc.
        self.agent.execute_command(request.data)
        response.success = True
        response.message = f"Command '{request.data}' executed for agent {self.agent_name}"
        return response
    

def main(args=None):
    rclpy.init(args=args)
    
    demo_trainer_node = DemoRLTrainer()
    executor = MultiThreadedExecutor()
    executor.add_node(demo_trainer_node)
    # Training loop
    training_thread = threading.Thread(target=demo_trainer_node.agent.train_loop)
    training_thread.start()
    
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    
    demo_trainer_node.destroy_node()
    rclpy.shutdown()
