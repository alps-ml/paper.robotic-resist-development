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
                "0.0499679",
                "--y",
                "0.0112094",
                "--z",
                "-0.0444582",
                "--qx",
                "0.706213",
                "--qy",
                "-0.681411",
                "--qz",
                "0.130662",
                "--qw",
                "-0.140959",
                # "--roll",
                # "0.0226994",
                # "--pitch",
                # "2.75541",
                # "--yaw",
                # "-1.61098",
            ],
        ),
    ]
    return LaunchDescription(nodes)
