cmake_minimum_required(VERSION 3.22)
project(rrtc_motion_planner LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall 
                      -Wextra
                      )
                      #-Wpedantic
endif()

# Common cmake code applied to all moveit packages
find_package(moveit_common REQUIRED)
moveit_package()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(moveit_core REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rsl REQUIRED)
find_package(trajectory_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)

find_package(rrtconnect_vendor QUIET)
find_package(rrtconnect REQUIRED)

set(THIS_PACKAGE_INCLUDE_DEPENDS 
      moveit_core rclcpp rsl 
      trajectory_msgs
      visualization_msgs
      rrtconnect_vendor rrtconnect
    )

add_library(rrtc_motion_planner SHARED
  src/rrtc_planner.cpp
  src/rrtc_utils.cpp)

target_include_directories(rrtc_motion_planner
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

set_target_properties(rrtc_motion_planner
                      PROPERTIES VERSION "${rrtc_motion_planner_VERSION}")

ament_target_dependencies(rrtc_motion_planner 
  PUBLIC
    ${THIS_PACKAGE_INCLUDE_DEPENDS})

target_link_libraries(rrtc_motion_planner
  PUBLIC
      rrtconnect::rrtconnect
)

install(
  TARGETS rrtc_motion_planner
  EXPORT rrtc_motion_plannerTargets
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
)

install(
  DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/include/${PROJECT_NAME}/
  DESTINATION include/${PROJECT_NAME}
)

ament_export_targets(rrtc_motion_plannerTargets HAS_LIBRARY_TARGET)
ament_export_include_directories(include)
ament_export_dependencies(${THIS_PACKAGE_INCLUDE_DEPENDS})

ament_package()
