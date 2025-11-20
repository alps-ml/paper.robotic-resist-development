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
                "0.055799",
                "--y",
                "0.0120742",
                "--z",
                "-0.0478456",
                "--qx",
                "0.704395",
                "--qy",
                "-0.683984",
                "--qz",
                "0.127445",
                "--qw",
                "-0.140541",
                # "--roll",
                # "0.0254827",
                # "--pitch",
                # "2.76065",
                # "--yaw",
                # "-1.60511",
            ],
        ),
    ]
    return LaunchDescription(nodes)
