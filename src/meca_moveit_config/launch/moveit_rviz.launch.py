import os
from launch import LaunchDescription
from launch_ros.actions import Node

from ament_index_python.packages import get_package_share_directory

from moveit_configs_utils import MoveItConfigsBuilder
from moveit_configs_utils.launches import generate_move_group_launch, generate_moveit_rviz_launch

def generate_launch_description():
    pkg_share = get_package_share_directory('meca_moveit_config')
    
    # This should match the robot name used in your SRDF <robot name="...">
    # and common config file naming conventions if not overridden.
    robot_name = "meca_500_r3" 

    urdf_file = os.path.join(pkg_share, "config", "meca_500_r3_tweezer.urdf")
    srdf_file = os.path.join(pkg_share, "config", "meca_500_r3.srdf")
    
    moveit_config_builder = MoveItConfigsBuilder(
                                robot_name=robot_name, 
                                package_name="meca_moveit_config" 
                            )
    
    # Explicitly set URDF and SRDF as their names dont match the 
    # <robot_name> convention
    moveit_config_builder.robot_description(file_path=urdf_file)
    moveit_config_builder.robot_description_semantic(file_path=srdf_file)

    # Let the builder auto-load ompl_planning.yaml, kinematics.yaml, 
    # joint_limits.yaml, controllers.yaml
    # from the 'config/' directory of 'meca_moveit_config'
    moveit_configs = moveit_config_builder.to_moveit_configs()

    # --- Robot State Publisher ---
    rsp_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[moveit_configs.robot_description] 
    )

    # --- MoveGroup Node using the helper ---
    move_group_launch_actions = generate_move_group_launch(moveit_configs)

    # --- RViz Node using the helper ---
    rviz_launch_actions = generate_moveit_rviz_launch(moveit_configs)

    # --- Scene Publisher ---
    scene_pub = Node(
        package='meca_moveit_config', 
        executable='scene_publisher',
        name='scene_publisher',
        output='screen',
        parameters=[{'env_yaml': os.path.join(pkg_share, 'config', 'environment.yaml')}]
    )

    # Build launch actions list conditionally
    launch_actions = [rsp_node, 
                      move_group_launch_actions,
                      rviz_launch_actions,
                      scene_pub]
    ld = LaunchDescription(launch_actions)
    return ld