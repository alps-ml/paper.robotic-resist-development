#!/usr/bin/env python3

import os
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    urdf_file = os.path.join(
        get_package_share_directory('mecademic_description'),
        'urdf',
        'meca.urdf')
        
    with open(urdf_file, 'r') as infp:
        robot_description_content = infp.read()
        
    rviz_config_file = os.path.join(
        get_package_share_directory('mecademic_description'),
        '.rviz2',
        'meca500_20250321.rviz')
        
    return LaunchDescription([
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            output='screen',
            parameters=[{'robot_description': robot_description_content}]
        ),
        Node(
            package = 'joint_state_publisher_gui',
            executable = 'joint_state_publisher_gui',
            output='screen'),
        Node(
            package = 'rviz2',
            executable = 'rviz2',
            name='rviz2',
            output='screen',
            arguments=['-d', rviz_config_file]),    
        ])
