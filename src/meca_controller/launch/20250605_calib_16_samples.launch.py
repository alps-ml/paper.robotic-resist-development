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
                "0.0569852",
                "--y",
                "0.0109603",
                "--z",
                "-0.0471076",
                "--qx",
                "0.70127",
                "--qy",
                "-0.68718",
                "--qz",
                "0.125112",
                "--qw",
                "-0.142656",
                # "--roll",
                # "0.0303059",
                # "--pitch",
                # "2.76093",
                # "--yaw",
                # "-1.59693",
            ],
        ),
    ]
    return LaunchDescription(nodes)
