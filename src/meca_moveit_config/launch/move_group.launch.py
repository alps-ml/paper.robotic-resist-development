import os
import yaml
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from moveit_configs_utils import MoveItConfigsBuilder
from moveit_configs_utils.launches import generate_move_group_launch

def generate_launch_description():
    # Build the MoveIt configuration using your package and robot name
    moveit_config = MoveItConfigsBuilder("meca_500_r3", package_name="meca_moveit_config").to_moveit_configs()

    # Get the package share directory for meca_moveit_config
    pkg_share = get_package_share_directory('meca_moveit_config')
    
    # Load the URDF file
    urdf_file = os.path.join(pkg_share, "config", "meca_500_r3_tweezer.urdf")
    with open(urdf_file, 'r') as f:
        robot_description = f.read()

    # Create the robot_state_publisher node with your URDF
    rsp_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{'robot_description': robot_description}]
    )

    # Load kinematics parameters from the kinematics.yaml file
    kinematics_file = os.path.join(pkg_share, "config", "kinematics.yaml")
    with open(kinematics_file, 'r') as kf:
        kinematics_params = yaml.safe_load(kf)
        
    # Generate the move_group launch description (which includes the move_group node)
    move_group_ld = generate_move_group_launch(moveit_config)

    # Create a new launch description that will include only the move_group node and the rsp_node
    ld = LaunchDescription()
    ld.add_action(rsp_node)
    
    # The generated move_group_ld is itself a LaunchDescription; we iterate over its entities
    for entity in move_group_ld.entities:
        if hasattr(entity, 'name') and entity.name == "move_group":
            # Define our custom parameters we want to enforce
            custom_params = {
                'robot_description_planning': kinematics_params
            }
            # If entity.parameters is not defined, create a list with our custom_params.
            if entity.parameters is None:
                entity.parameters = [custom_params]
            else:
                # Update each existing dictionary in the parameters list with our custom parameters.
                for i in range(len(entity.parameters)):
                    entity.parameters[i].update(custom_params)
        ld.add_action(entity)

    return ld
