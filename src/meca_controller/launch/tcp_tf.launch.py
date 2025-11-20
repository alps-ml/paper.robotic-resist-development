from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description() -> LaunchDescription:
    # Translation (meters) from meca_axis_6_link to your tweezer tip point (TRF)
    tcp_x = "0.0172"    #0.0164 offset in X of flange
    tcp_y = "0.0"    # offset in Y of flange
    tcp_z = "-0.0812"  # offset in Z of flange (parallel to tweezer tip)


    # Rotation (quaternion x,y,z,w) of TCP frame relative to meca_axis_6_link frame
    # Default: TCP axes aligned with meca_axis_6_link axes
    tcp_qx = "0.0"
    tcp_qy = "0.0"
    tcp_qz = "0.0"
    tcp_qw = "1.0"
    # If your tweezers are, for example, rotated 90 degrees around Y axis of flange:
    # tcp_qx = "0.0"; tcp_qy = "0.7071068"; tcp_qz = "0.0"; tcp_qw = "0.7071068";

    nodes = [
        Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            name='tf_EF_to_Tooling',
            output="log",
            arguments=[
                "--frame-id", "meca_axis_6_link",
                "--child-frame-id", "tweezer_tcp", # Tool Center Point frame name
                "--x", tcp_x, "--y", tcp_y, "--z", tcp_z,
                "--qx", tcp_qx, "--qy", tcp_qy, "--qz", tcp_qz, "--qw", tcp_qw,
            ],
        ),
    ]
    return LaunchDescription(nodes)