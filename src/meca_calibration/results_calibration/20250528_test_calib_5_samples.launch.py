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
                "0.0122124",
                "--y",
                "0.0380058",
                "--z",
                "-0.187563",
                "--qx",
                "0.704668",
                "--qy",
                "-0.68239",
                "--qz",
                "0.125981",
                "--qw",
                "-0.148041",
                # "--roll",
                # "0.039682",
                # "--pitch",
                # "2.75224",
                # "--yaw",
                # "-1.61074",
            ],
        ),
    ]
    return LaunchDescription(nodes)
