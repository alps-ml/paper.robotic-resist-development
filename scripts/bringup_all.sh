#!/usr/bin/env bash

LAUNCH_RVIZ=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --rviz)
            LAUNCH_RVIZ=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --rviz        Launch MoveIt with RViz"
            echo "  --help, -h   Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Cleanup: kill any previously started bringup nodes (but not the driver)
echo "Cleaning up any previously running bringup nodes..."
ps aux | grep -E 'realsense2_camera|calibPark|tcp_tf|moveit_rviz|chip_detection_node|rqt_image_view|rviz2' | grep -v grep | awk '{print $2, $11, $12, $13}'
ps aux | grep -E 'realsense2_camera|calibPark|tcp_tf|moveit_rviz|chip_detection_node|rqt_image_view|rviz2' | grep -v grep | awk '{print $2}' | xargs -r kill
sleep 2

# Source ROS2 and workspace
source ${ROS2_INSTALL:-"/opt/ros/jazzy/setup.bash"}
source ${ROS2_WS:-:".."}/venv/bin/activate
source ${ROS2_WS:-:".."}/install/setup.bash

ros2 launch realsense2_camera rs_launch.py \
    json_file_path:="$RFS_CONFIG_CAMERA_CALIBRATION" \
    depth_module.contrast:=24 \
    depth_module.enable_auto_exposure:=false \
    depth_module.brightness:=-14 \
    depth_module.exposure:=28685 \
    depth_module.gain:=16 \
    enable_infra1:=false \
    enable_infra2:=false \
    enable_depth:=false \
    enable_auto_white_balance:=false \
    publish_tf:=false &
CAM_PID=$!

# Launch MoveIt (always needed for TF tree, RViz controlled by parameter)
echo "Launching MoveIt..."
if [ "$LAUNCH_RVIZ" = true ]; then
    echo "Launching MoveIt with RViz..."
    ros2 launch meca_moveit_config moveit_rviz.launch.py 2>&1 | grep -v "because this Material does not exist in group General" &
else
    echo "Launching MoveIt without RViz..."
    ros2 launch meca_moveit_config moveit.launch.py &
fi
MOVEIT_PID=$!

ros2 launch meca_controller tcp_tf.launch.py &
TCP_TF_PID=$!

ros2 launch meca_controller 20250714_calibPark_9_samples.launch.py &
CALIB_TF_PID=$!

# Wait for TF and camera_info to be available before starting chip detection
echo "Waiting for /tf to be active..."
until ros2 topic info /tf 2>/dev/null | grep -q "Publisher count: [1-9]"; do
    sleep 0.5
done
echo "Topic '/tf' is available."

echo "Waiting for camera_info..."
until ros2 topic info /camera/camera/color/camera_info 2>/dev/null | grep -q "Publisher count: [1-9]"; do
    sleep 0.5
done
echo "Topic 'camera_info' is available."

ros2 run meca_perception chip_detection_node &
CHIP_DET_PID=$!

ros2 run rqt_image_view rqt_image_view &
RQT_PID=$!

# Save PIDs of all background jobs started by this script
jobs -p > bringup_pids.txt

echo "PIDs of background jobs saved to bringup_pids.txt"

cat << EOF

All nodes launched in the background.

- To stop all nodes, run:
    kill_bringup_nodes.sh
EOF
