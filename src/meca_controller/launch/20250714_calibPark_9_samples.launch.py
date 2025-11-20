""" Static transform publisher acquired via MoveIt 2 hand-eye calibration """
""" EYE-IN-HAND: meca_axis_5_link -> camera_color_optical_frame """
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    nodes = [
        Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            name='tf_axis5_to_camera',
            output="log",
            arguments=[
                "--frame-id",
                "meca_axis_5_link",
                "--child-frame-id",
                "camera_color_optical_frame",
                "--x",
                "0.056165",
                "--y",
                "0.0105088",
                "--z",
                "-0.0482255",
                "--qx",
                "0.705904",
                "--qy",
                "-0.684219",
                "--qz",
                "0.122234",
                "--qw",
                "-0.136391",
                # "--roll",
                # "0.0271",
                # "--pitch",
                # "2.77417",
                # "--yaw",
                # "-1.60703",
            ],
        ),
    ]
    return LaunchDescription(nodes)
