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
                "0.060934",
                "--y",
                "0.0110846",
                "--z",
                "-0.0637795",
                "--qx",
                "0.694191",
                "--qy",
                "-0.691537",
                "--qz",
                "0.13248",
                "--qw",
                "-0.149411",
                # "--roll",
                # "0.0263019",
                # "--pitch",
                # "2.74033",
                # "--yaw",
                # "-1.57998",
            ],
        ),
    ]
    return LaunchDescription(nodes)
