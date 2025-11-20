""" Static transform publisher acquired via MoveIt 2 hand-eye calibration """
""" EYE-IN-HAND: meca_axis_5_link -> camera_color_optical_frame """
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    nodes = [
        Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            output="log",
            arguments=[
                "--frame-id",
                "meca_axis_5_link",
                "--child-frame-id",
                "camera_color_optical_frame",
                "--x",
                "0.0556486",
                "--y",
                "0.0127098",
                "--z",
                "-0.0470984",
                "--qx",
                "0.705083",
                "--qy",
                "-0.683167",
                "--qz",
                "0.128321",
                "--qw",
                "-0.140265",
                # "--roll",
                # "0.024214",
                # "--pitch",
                # "2.75978",
                # "--yaw",
                # "-1.60705",
            ],
        ),
    ]
    return LaunchDescription(nodes)
