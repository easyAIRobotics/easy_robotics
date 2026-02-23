cmake_minimum_required(VERSION 3.8)
project(easy_simulation)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

find_package(gz_sim_vendor REQUIRED)
find_package(gz-sim REQUIRED)

find_package(gz_plugin_vendor REQUIRED)
find_package(gz-plugin REQUIRED)

find_package(gz_common_vendor REQUIRED)
find_package(gz-common REQUIRED)

include_directories(
  include
)

# Install Python executable scripts
install(
  PROGRAMS
    scripts/box_spawner.py
  DESTINATION lib/${PROJECT_NAME}
)

install(
  DIRECTORY config launch urdf worlds
  DESTINATION share/${PROJECT_NAME}
)

# Install Gazebo plugins
add_library(easy_simulation_plugins SHARED
  src/suction_gripper_gazebo.cpp
)
set_property(TARGET easy_simulation_plugins PROPERTY CXX_STANDARD 17)

target_link_libraries(easy_simulation_plugins
  PUBLIC
  gz-sim::gz-sim
  gz-plugin::register
)

target_include_directories(easy_simulation_plugins PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include/${PROJECT_NAME}>
)

# Install headers
install(DIRECTORY include/
  DESTINATION include/${PROJECT_NAME}
)

install(TARGETS easy_simulation_plugins
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_export_include_directories(include)
ament_export_libraries(easy_simulation_plugins)

ament_package()
