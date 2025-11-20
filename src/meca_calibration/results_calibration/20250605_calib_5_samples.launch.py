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
                "0.0513447",
                "--y",
                "0.011665",
                "--z",
                "-0.0457583",
                "--qx",
                "0.708723",
                "--qy",
                "-0.679344",
                "--qz",
                "0.127805",
                "--qw",
                "-0.140956",
                # "--roll",
                # "0.0281843",
                # "--pitch",
                # "2.75971",
                # "--yaw",
                # "-1.61857",
            ],
        ),
    ]
    return LaunchDescription(nodes)
