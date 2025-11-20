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
                "-0.0291416",
                "--y",
                "-0.117895",
                "--z",
                "-0.158259",
                "--qx",
                "-0.00551",
                "--qy",
                "0.699005",
                "--qz",
                "0.163457",
                "--qw",
                "0.696163",
                # "--roll",
                # "1.66671",
                # "--pitch",
                # "1.81036",
                # "--yaw",
                # "-1.44125",
            ],
        ),
    ]
    return LaunchDescription(nodes)
