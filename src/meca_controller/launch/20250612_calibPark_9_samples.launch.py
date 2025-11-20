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
                "0.055988",
                "--y",
                "0.0108364",
                "--z",
                "-0.0475921",
                "--qx",
                "0.701982",
                "--qy",
                "-0.686417",
                "--qz",
                "0.125271",
                "--qw",
                "-0.142691",
                # "--roll",
                # "0.0305517",
                # "--pitch",
                # "2.76068",
                # "--yaw",
                # "-1.59911",
            ],
        ),
    ]
    return LaunchDescription(nodes)
