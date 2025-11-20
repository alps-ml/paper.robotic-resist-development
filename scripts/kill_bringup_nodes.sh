#!/usr/bin/env bash

# Helper script to kill only the bringup nodes (not the driver)

PATTERNS=(
  'realsense2_camera'
  'calibPark'
  'tcp_tf'
  'moveit_rviz'
  'chip_detection_node'
  'rviz2'
  'robot_state_publisher'  # Add this to catch ghost processes
  'static_transform_publisher'  # Add this for TF publishers
  'scene_publisher'  # Add this for MoveIt scene publisher
  'move_group'  # Add this for MoveIt move_group
  'joint_state_publisher'  # Add this for joint state publishers
)

echo "Killing bringup nodes..."

for pattern in "${PATTERNS[@]}"; do
  echo "Killing processes matching: $pattern"
  pkill -f "$pattern"
done

echo "Force killing GUI processes (SIGKILL) to ensure they are closed..."
pkill -9 -f 'rqt_image_view'
pkill -9 -f 'rviz2'

echo "Cleaning up any remaining temporary files..."
rm -f /tmp/launch_params_* 2>/dev/null || true

echo "All bringup nodes killed (except the driver and any unrelated nodes)."
echo ""
echo "Remaining ROS2 processes:"
ps aux | grep -E 'ros2|meca_driver' | grep -v grep || echo "No ROS2 processes found." 