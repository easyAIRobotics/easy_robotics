cmake_minimum_required(VERSION 3.22)
project(easy_kinova_g3l_moveit_config)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(moveit_core REQUIRED)
find_package(moveit_ros_planning REQUIRED)
find_package(moveit_ros_planning_interface REQUIRED)
find_package(tf2_eigen REQUIRED)
find_package(easy_interfaces REQUIRED)

ament_package()

add_executable(moveit_executor_node
  src/moveit_executor_node.cpp
)

ament_target_dependencies(moveit_executor_node
  rclcpp
  moveit_core
  moveit_ros_planning
  moveit_ros_planning_interface
  tf2_eigen
  easy_interfaces
)

install(TARGETS moveit_executor_node
  DESTINATION lib/${PROJECT_NAME}
)

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/launch")
  install(
    DIRECTORY launch
    DESTINATION share/${PROJECT_NAME}
    PATTERN "setup_assistant.launch" EXCLUDE)
endif()

install(DIRECTORY config DESTINATION share/${PROJECT_NAME})
install(FILES .setup_assistant DESTINATION share/${PROJECT_NAME})
