cmake_minimum_required(VERSION 3.22)
project(moveit_linqs_rrtc LANGUAGES CXX)

# Common cmake code applied to all moveit packages
find_package(moveit_common REQUIRED)
moveit_package()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(moveit_core REQUIRED)
find_package(pluginlib REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rsl REQUIRED)

find_package(rrtc_motion_planner REQUIRED)

set(THIS_PACKAGE_INCLUDE_DEPENDS 
      moveit_core rclcpp rsl pluginlib
      rrtc_motion_planner
    )

include_directories(include)

add_library(rrtc_motion_interface SHARED src/rrtc_interface.cpp
                                          src/rrtc_planning_context.cpp)
set_target_properties(rrtc_motion_interface
                      PROPERTIES VERSION "${moveit_linqs_rrtc_VERSION}")

ament_target_dependencies(rrtc_motion_interface PUBLIC ${THIS_PACKAGE_INCLUDE_DEPENDS})

add_library(rrtc_motion_planner_plugin SHARED src/rrtc_plugin.cpp)
ament_target_dependencies(rrtc_motion_planner_plugin PUBLIC ${THIS_PACKAGE_INCLUDE_DEPENDS})
target_link_libraries(rrtc_motion_planner_plugin PUBLIC rrtc_motion_interface)


if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  target_compile_options(rrtc_motion_interface 
    PRIVATE
      -Wno-old-style-cast
      -Wno-unused-parameter
      -Wno-pedantic
      -Wno-redundant-decls
    INTERFACE 
      -Wno-unused-parameter)
endif()

install(
  TARGETS rrtc_motion_interface rrtc_motion_planner_plugin
  EXPORT moveit_linqs_rrtcTargets
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin)

ament_export_targets(moveit_linqs_rrtcTargets HAS_LIBRARY_TARGET)
ament_export_dependencies(${THIS_PACKAGE_INCLUDE_DEPENDS})


pluginlib_export_plugin_description_file(moveit_core
                                         plugins/rrtc_interface_plugin_description.xml)
ament_package()
