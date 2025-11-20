cmake_minimum_required(VERSION 3.8)
project(moveit_linqs_historical LANGUAGES CXX)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(pluginlib REQUIRED)
find_package(moveit_core REQUIRED)
find_package(moveit_ros_planning REQUIRED)

find_package(chomp_motion_planner REQUIRED)
find_package(moveit_planners_stomp REQUIRED)

set(THIS_PACKAGE_INCLUDE_DEPENDS
  moveit_core
  moveit_ros_planning
  pluginlib
  rclcpp
)

include_directories(include)

# CHOMP Smoothing
add_library(moveit_linqs_historical_chomp_optimizer_adapter SHARED 
            src/chomp_optimizer_adapter.cpp)
set_target_properties(moveit_linqs_historical_chomp_optimizer_adapter PROPERTIES 
                      VERSION "${moveit_chomp_optimizer_adapter_VERSION}")
ament_target_dependencies(moveit_linqs_historical_chomp_optimizer_adapter PUBLIC 
                          chomp_motion_planner 
                          ${THIS_PACKAGE_INCLUDE_DEPENDS})

install(
  TARGETS moveit_linqs_historical_chomp_optimizer_adapter
  EXPORT moveit_linqs_historical_chomp_optimizer_adapterTargets
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
)
ament_export_targets(moveit_linqs_historical_chomp_optimizer_adapterTargets HAS_LIBRARY_TARGET)
pluginlib_export_plugin_description_file(moveit_core chomp_optimizer_adapter_plugin_description.xml)

# STOMP Smoothing
add_library(moveit_linqs_historical_stomp_smoothing_adapter SHARED 
            src/stomp_smoothing_adapter.cpp)
set_target_properties(moveit_linqs_historical_stomp_smoothing_adapter PROPERTIES 
                      VERSION "${moveit_stomp_smoothing_adapter_VERSION}")
ament_target_dependencies(moveit_linqs_historical_stomp_smoothing_adapter PUBLIC
                          moveit_planners_stomp
                          ${THIS_PACKAGE_INCLUDE_DEPENDS})

install(
  TARGETS moveit_linqs_historical_stomp_smoothing_adapter
  EXPORT moveit_linqs_historical_stomp_smoothing_adapterTargets
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
)

ament_export_targets(moveit_linqs_historical_stomp_smoothing_adapterTargets HAS_LIBRARY_TARGET)
pluginlib_export_plugin_description_file(moveit_core stomp_smoothing_adapter_plugin_description.xml)

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

ament_package()
