search_mode=OPTIMIZE_MAX_JOINT
srdf_filename=meca_500_r3.srdf
robot_name_in_srdf=meca_500_r3
moveit_config_pkg=meca_moveit_config
robot_name=meca_500_r3
planning_group_name=meca_arm
ikfast_plugin_pkg=meca_500_r3_meca_arm_ikfast_plugin
base_link_name=meca_base_link
eef_link_name=meca_axis_6_link
ikfast_output_path=/home/gyger/projects/robotsforscience/src/ros_ws/src/meca_500_r3_meca_arm_ikfast_plugin/src/meca_500_r3_meca_arm_ikfast_solver.cpp

ros2 run moveit_kinematics create_ikfast_moveit_plugin.py\
  --search_mode=$search_mode\
  --srdf_filename=$srdf_filename\
  --robot_name_in_srdf=$robot_name_in_srdf\
  --moveit_config_pkg=$moveit_config_pkg\
  $robot_name\
  $planning_group_name\
  $ikfast_plugin_pkg\
  $base_link_name\
  $eef_link_name\
  $ikfast_output_path
