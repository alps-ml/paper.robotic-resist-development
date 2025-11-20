"""
meca_driver.py

This node establishes a connection to whichever robot is specified in the arguments;
It receives commands over service requests from other nodes (e.g. meca_control.py) and carries
them out by communicating with the robot using the mecademicpy library. 
It also publishes the robot's state (status, joints, pose) over a topic, to which other nodes can subscribe
to learn the current joint angles, etc.

The node is designed to be run with a namespace, so that it can be started with:
ros2 run meca_controller meca_driver --ros-args -r __ns:=/robot1

NOTE: This code was adapted from Jessica Myers ROS2 project (https://github.com/myersjm/mecademic-ros2) 
and the Mecademic ROS 1 driver node at the Mecademic repo:  (https://github.com/Mecademic/ROS/tree/5c471e98a834b68503c93bd4c6a4719c32e3e491).

LICENSE: MIT
"""

import math
import time

import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup

from geometry_msgs.msg import Pose
from sensor_msgs.msg import JointState
from control_msgs.action import FollowJointTrajectory

from custom_interfaces.srv import (MoveJoints, GoToPose, MoveGripper, SetBlending, 
                                   SetGripperForce, SetGripperVel, SetJointVel, SetJointAcc, 
                                   MoveLin, MoveLinRelWrf, MovePoseEuler, GetRtJointTorq, 
                                   ClearMotion, WaitIdle, Home)
from custom_interfaces.msg import RobotStatus, GripperStatus
from custom_interfaces.action import FollowCartesianTrajectory

from scipy.spatial.transform import Rotation as R

import mecademicpy.robot as mdr
from mecademicpy._robot_base import disconnect_on_exception_decorator

# Monkey Patch mecademicpy to add GetRtJointTorq.
@disconnect_on_exception_decorator
def _GetRtJointTorq(self,
                include_timestamp: bool = False,
                synchronous_update: bool = False,
                timeout: float = None):
    if include_timestamp:
        return self._robot_rt_data.rt_joint_torq
    else:
        return self._robot_rt_data.rt_joint_torq.data

mdr.Robot.GetRtJointTorq = _GetRtJointTorq


from meca_controller.meca_settings import ROBOT1


class Meca_Driver(Node):
    def __init__(self, robot_config: dict):
        super().__init__("meca_driver")
        self.get_logger().info('init: Meca Driver Node starting up.')
        
        # FIXME this should probably be set differently if there is another way to configure settings in ROS2.
        self._ROBOT_CONFIG = robot_config
        
        self._robot_hardware_callback_group = MutuallyExclusiveCallbackGroup()
        self._initialize_publishers()
        self._initialize_service_handlers()
        self._initialize_action_servers()

        self._setup_robot_hardware()

    def _initialize_publishers(self):
        """ Initialize publishers.

        NOTE that when this node is run with namespace provided, the service will be '{namespace}/pose/'
        Otherwise it needs to be started with a slash to '/joint_states'.
        """
        self.get_logger().debug('init: Initializing publishers.')
        self.joint_publisher = self.create_publisher(JointState, "/joint_states", 1)
        self.pose_publisher = self.create_publisher(Pose, "pose", 1)
        self.robot_status_publisher = self.create_publisher(RobotStatus, "robot_status", 1)
        self.gripper_status_publisher = self.create_publisher(GripperStatus, "gripper_status", 1)

    def _initialize_service_handlers(self):
        """ Set up services.
            Currently all the services but wait_idle and home are non-blocking, so the movement is not finished when the service returns.
            NOTE that when this node is run with namespace provided, the service will be '{namespace}/move_joints/'
        """
        self.get_logger().debug('init: Initializing service handlers.')
        self.srv_set_blending = self.create_service(SetBlending, 'set_blending', self.set_blending_callback, callback_group=self._robot_hardware_callback_group)
        self.srv_set_gripper_force = self.create_service(SetGripperForce, 'set_gripper_force', self.set_gripper_force_callback, callback_group=self._robot_hardware_callback_group)
        self.srv_set_gripper_vel = self.create_service(SetGripperVel, 'set_gripper_vel', self.set_gripper_vel_callback, callback_group=self._robot_hardware_callback_group)
        self.srv_set_joint_vel = self.create_service(SetJointVel, 'set_joint_vel', self.set_joint_vel_callback, callback_group=self._robot_hardware_callback_group)
        self.srv_set_joint_acc = self.create_service(SetJointAcc, 'set_joint_acc', self.set_joint_acc_callback, callback_group=self._robot_hardware_callback_group)
        self.srv_clear_motion = self.create_service(ClearMotion, 'clear_motion', self.clear_motion_callback, callback_group=self._robot_hardware_callback_group)
        self.srv_get_rt_joint_torq = self.create_service(GetRtJointTorq, 'get_rt_joint_torq',self.get_rt_joint_torq_callback, callback_group=self._robot_hardware_callback_group)

        self.srv_move_joints = self.create_service(MoveJoints, 'move_joints', self.move_joints_callback, callback_group=self._robot_hardware_callback_group)
        self.srv_move_pose = self.create_service(MovePoseEuler, 'move_pose', self.move_pose_callback, callback_group=self._robot_hardware_callback_group)
        self.srv_go_to_pose = self.create_service(GoToPose, 'go_to_pose', self.go_to_pose_callback, callback_group=self._robot_hardware_callback_group)

        self.srv_move_lin_rel_wrf = self.create_service(MoveLinRelWrf, 'move_lin_rel_wrf', self.move_lin_rel_wrf_callback, callback_group=self._robot_hardware_callback_group)
        self.srv_move_lin = self.create_service(MoveLin, 'move_lin', self.move_lin_callback, callback_group=self._robot_hardware_callback_group)
        self.srv_move_gripper = self.create_service(MoveGripper, 'move_gripper', self.move_gripper_callback, callback_group=self._robot_hardware_callback_group)
        self.srv_wait_idle = self.create_service(WaitIdle, 'wait_idle', self.wait_idle_callback, callback_group=self._robot_hardware_callback_group)
        self.srv_home = self.create_service(Home, 'home', self.home_callback, callback_group=self._robot_hardware_callback_group)

    def _initialize_action_servers(self):
        """ Initialize action servers.
        """
        self.get_logger().debug('init: Initializing action servers.')
        # Action server so MoveIt can execute trajectories automatically:
        self._joint_traj_action_server = ActionServer(
            self,
            FollowJointTrajectory,
            'meca_arm_controller/follow_joint_trajectory',
            execute_callback=self._execute_joint_trajectory_callback,
            callback_group=self._robot_hardware_callback_group
        )

        # Action server so Cartesian trajectories can be sent.
        self._cart_traj_action_server = ActionServer(
            self,
            FollowCartesianTrajectory,
            'meca_arm_controller/follow_cartesian_trajectory',
            execute_callback=self._execute_cartesian_trajectory_callback,
            callback_group=self._robot_hardware_callback_group
        )
        
    def _setup_robot_hardware(self):
        """ Initializes the connection to the Mecademic robot hardware.
        """
        self.get_logger().info('init: Robot (MECA 500) starting up.')
        self.robot = mdr.Robot() # mecademic py api interface

        # Setup callbacks:
        self._robot_callbacks = mdr.RobotCallbacks()
        # In the on_end_of_cycle callback, all real-time data is consistent.
        self._robot_callbacks.on_end_of_cycle = self.on_end_of_cycle_callback
        self.robot.RegisterCallbacks(self._robot_callbacks, run_callbacks_in_separate_thread=True)

        self.robot.Connect(address=self._ROBOT_CONFIG['ip'], disconnect_on_exception=False)
        
        # Start robot:
        self.robot.ActivateRobot()
        
        monitoring_topics = [
            mdr.MxRobotStatusCode.MX_ST_RT_JOINT_POS,
            mdr.MxRobotStatusCode.MX_ST_RT_JOINT_VEL,
            mdr.MxRobotStatusCode.MX_ST_RT_JOINT_TORQ,
            mdr.MxRobotStatusCode.MX_ST_RT_CART_POS
        ]
        self.robot.SetRealTimeMonitoring(*monitoring_topics) # Limit topics to relevant information.
        #self.robot.SetRealTimeMonitoring('all')

        self.robot.Home()
        self.robot.WaitIdle(timeout=60) # wait for the robot to finish its initial movement

        self.get_logger().info('init: Robot ready. Startup finished.')

        # Set monitoring interval for real time data. The callback then published this data in ROS2 network
        try:
            # This command can fail if interval is invalid
            self.robot.SetMonitoringInterval(self._ROBOT_CONFIG['monitoring_interval']) 
            self.get_logger().info(f"init: Monitoring interval set to {self._ROBOT_CONFIG['monitoring_interval']}s")
        except Exception as e:
            self.get_logger().error(f"init: Failed to set monitoring interval: {e}")

    @property
    def namespace(self):
        """ Returns namespace of this node.
        """
        self.get_namespace()
    
    @property
    def is_in_error(self):
        """ Returns if the robot is in an error state (True for error)
        """
        try:
            status = self.robot.GetStatusRobot()
            return status.error_status
        except Exception as e:
            self.get_logger().error(f"is_in_error: Failed to get robot status: {e}")
            return False

    def stop(self, deactivate=True):
        """ Shuts down the robots, deactivating and closing the socket connection (disconnecting).
        """
        try:
            self.robot.WaitIdle(60)
            if deactivate:
                self.robot.DeactivateRobot()
            self.robot.Disconnect()
            self.get_logger().info('stop: Robot stopped')
        except mdr.DisconnectError:
            self.stop_after_disconnect_error()

    def stop_after_disconnect_error(self):
        """ Reconnects to the robot to properly shutdown the robot.

        Purpose: the DisconnectError is raised by Mecademicpy when an exception occurs, even if that exception is handled. 
                 In this node, CTRL-C is used to shut down the node and robots, so we do not want things to end without the robot being
                 properly deactivated and disconnected. 
                 
                 This function will reconnect for the purpose of properly shutting things down.
        """
        self.get_logger().debug('stop_after_disconnect_error: Reconnecting to robot to properly shut down.')
        self.robot.Connect(address=self._ROBOT_CONFIG['ip'])
        
        # Reset the error, if there is any, and resume robot motion if it 
        #    is paused (it will be after a DisconnectError):
        self.robot.ResetError() # just in case
        self.robot.ResumeMotion()
        self.stop()

    def clear_error(self):
        """ ResetError & ResumeMotion: If the robot is in an error state, this can be called to reset the error and resume motion.
        """
        self.get_logger().info('clear_error: Reset error requested.')
        self.robot.ResetError()
        self.robot.ResumeMotion()

    def on_end_of_cycle_callback(self):
        """ Publishes the robot's status, gripper status, and joint state (position and velocity) every MONITORING_INTERVAL seconds.

            NOTE: This callback is called by the mecademicpy library in a separate thread owned by the library.
        """
        try: 
            # === Robot Status Publishing ===
            robot_status = self.robot.GetStatusRobot(synchronous_update=False)
            gripper_status = self.robot.GetStatusGripper(synchronous_update=False)

            robot_status_msg = RobotStatus()
            robot_status_msg.activation_state = robot_status.activation_state
            robot_status_msg.brakes_engaged = robot_status.brakes_engaged
            robot_status_msg.simulation_mode = robot_status.simulation_mode

            gripper_status_msg = GripperStatus()
            gripper_status_msg.target_pos_reached = gripper_status.target_pos_reached

            self.robot_status_publisher.publish(robot_status_msg)
            self.gripper_status_publisher.publish(gripper_status_msg)

            # === Joint and Pose Publishing ===

            # In this callback, the realtime data is always a valid configuration.
            data = self.robot.GetRobotRtData(synchronous_update=False) 
            if not data: 
                self.get_logger().warn("Data update: GetRobotRtData returned empty data.")
                return

            joints = JointState()
            joints.header.stamp = self.get_clock().now().to_msg()
            joints.header.frame_id = "meca_base_link"

            joints.name = [ 
                "meca_axis_1_joint", "meca_axis_2_joint", "meca_axis_3_joint",
                "meca_axis_4_joint", "meca_axis_5_joint", "meca_axis_6_joint",
                "meca_gripper_finger1_joint"
            ] 
            
            try:
                arm_positions = [math.radians(x) for x in data.rt_joint_pos.data[0:6]]
                gripper_pos = 0.0
                joints.position = arm_positions + [gripper_pos] 
            except Exception as e:
                self.get_logger().warn("Data update: Could not get joint positions from RtData.")
                joints.position = [0.0] * 7 # Default if unavailable

            try:
                arm_velocities = data.rt_joint_vel.data[0:6]
                gripper_vel = 0.0
                joints.velocity = arm_velocities + [gripper_vel]
            except Exception as e:
                self.get_logger().warn("Data update: Could not get joint velocities from RtData.")
                joints.velocity = [0.0] * 7 # Default if unavailable

            self.joint_publisher.publish(joints)
            
            try:
                pose = Pose()
                raw_cart_pos = data.rt_cart_pos.data
            
                if len(raw_cart_pos) >= 3:
                    pose.position.x = float(raw_cart_pos[0])
                    pose.position.y = float(raw_cart_pos[1])
                    pose.position.z = float(raw_cart_pos[2])
                    
                    # Handle orientation (still using Euler angles in x,y,z as per original)
                    # FIXME: SG, I feel that should be ideally quaternions
                    if len(raw_cart_pos) >= 6: #FIXME: Somehow this makes no sense to me, why should there be less?
                        pose.orientation.x = float(raw_cart_pos[3]) # Alpha
                        pose.orientation.y = float(raw_cart_pos[4]) # Beta
                        pose.orientation.z = float(raw_cart_pos[5]) # Gamma
                        pose.orientation.w = 0.0 # Set W=0 when using Euler angles in x,y,z
                    else:
                        self.get_logger().warn("Data update: Received fewer than 6 values for Cartesian pose, orientation incomplete.")
                        pose.orientation.x = 0.0
                        pose.orientation.y = 0.0
                        pose.orientation.z = 0.0
                        pose.orientation.w = 1.0 # Identity quaternion is maybe safer default
                else:
                    self.get_logger().warn(f"Data update: Received incomplete rt_cart_pos data (len={len(raw_cart_pos)}).")
                    raise Exception("Data update: Received incomplete rt_cart_pos data.")
            except:
                self.get_logger().warn("Data update: Could not get Cartesian pose from RtData.")
                return

            self.pose_publisher.publish(pose) 

        except mdr.CommunicationError as e:
             self.get_logger().error(f"Data update: Communication error in timer callback: {e}")
        except Exception as e:
             self.get_logger().error(f"Data update: Unexpected error in on_end_of_cycle_callback: {e}")

    def wait_idle_callback(self, request, response):
        """ WaitIdle: Wait for the robot to be idle.

        Request inputs:
            - timeout_sec: timeout in seconds.
        
        Response:
            - success [bool]: True if the robot is idle, False otherwise.
            - error_message [string]: Error message if the wait failed.
        """
        self.get_logger().info(f"WaitIdle: Request received (timeout: {request.timeout_sec}s)")
        try:
            # The mecademicpy WaitIdle() function blocks until the robot is physically idle
            # or the timeout (in seconds) is reached on the robot side.
            self.robot.WaitIdle(timeout=request.timeout_sec) 
            response.success = True
            response.error_message = ""
            self.get_logger().info("WaitIdle: Robot is now idle.")
        except mdr.TimeoutException: # Catch mecademicpy timeout
            self.get_logger().error(f"WaitIdle: Robot did not become idle within the robot's {request.timeout_sec}s timeout.")
            response.success = False
            response.error_message = "WaitIdle timed out on robot"
        except Exception as e:
            self.get_logger().error(f"WaitIdle service failed with an unexpected error: {e}")
            response.success = False
            response.error_message = str(e)
        return response

    def clear_motion_callback(self, request, response):
        """ ClearMotion: This command stops the robot movement by decelerating. The rest of the trajectory is deleted.

        Request inputs:
            - None
        
        Response:
            - success [bool]: True if the motion buffer was cleared, False otherwise.
            - error_message [string]: Error message if the clear motion failed.
        """
        self.get_logger().info("ClearMotion: received request")
        
        try:
            self.robot.ClearMotion()
            self.robot.ResumeMotion()
            response.success = True
            response.message = "Motion buffer cleared successfully"
            self.get_logger().info("ClearMotion: Motion buffer cleared successfully.")
        except Exception as e:
            self.get_logger().error(f"ClearMotion: service failed: {e}")
            response.success = False
            response.message = f"Failed to clear motion buffer: {str(e)}"
        return response

    def home_callback(self, request, response):
        """ Home: Home the robot.
        
        Request inputs:
            - None
        
        Response:
            - success [bool]: True if the robot is homed, False otherwise.
            - error_message [string]: Error message if the home failed.
        """

        self.get_logger().debug("Home: received request")
        try:
            self.robot.Home()
            self.robot.WaitHomed(timeout=60)
            response.success = True
            response.error_message = "Robot homed successfully"
            self.get_logger().info("Home: Robot homed.")
        except mdr.TimeoutException:
            self.get_logger().error(f"Home: Robot did not become idle within the requests timeout.")
            response.success = False
            response.error_message = "Home timed out on robot"
        except Exception as e:
            self.get_logger().error(f"Home: service failed with an unexpected error: {e}")
            response.success = False
            response.error_message = str(e)
        return response

    def move_joints_callback(self, request, response):
        """ MoveJoints: Move the robot tool to a specified joint configuration.
        
        Request inputs:
            - float64[] requested_joint_angles
        
        Response: 
            - None (besides whatever is passed in)
        """
        desired_joint_angles = request.requested_joint_angles.tolist()
        self.get_logger().info(f'MoveJoints: moving to {desired_joint_angles}')

        self.robot.MoveJoints(*desired_joint_angles) # has no return value
        return response
    
    def go_to_pose_callback(self, request, response):
        """ MovePose: Move the robot tool to a specified pose.
        
        NOTE this has not yet been tested or incorporated with the motion planner. Use at your own risk.
        FIXME: Unclear whats the difference between this and MovePose

        Request inputs:
            - Pose msg: position and orientation information
        
        Response: None
        """
        self.get_logger().info(f'MovePose: moving to x={request.pose.position.x}, y={request.pose.position.y}, z={request.pose.position.z}, '
                               f'alpha={request.pose.orientation.x}, beta={request.pose.orientation.y}, gamma={request.pose.orientation.z}')
        self.robot.MovePose(x=request.pose.position.x, 
                            y=request.pose.position.y, 
                            z=request.pose.position.z,
                            alpha=request.pose.orientation.x, 
                            beta=request.pose.orientation.y, 
                            gamma=request.pose.orientation.z)
        return response

    def move_gripper_callback(self, request, response):
        """ MoveGripper: Move the gripper.
        
        Request inputs:
            - command [String]: {"open", "close", "pos"}
            - pos [float]: gripper position in mm in range [0, 30.0]
        
        Response:
            - error [bool]: True if error occurred, False otherwise.
        """
        try:
            status = self.robot.GetStatusRobot()
            self.get_logger().info(f"Idle check: {status}")
            self.robot.WaitIdle(timeout=60)
        except Exception as te:
            self.get_logger().warn(f"MoveGripper: WaitIdle timed out: {te} -> proceeding with gripper.")

        if request.command == "open":
            self.get_logger().info(f'MoveGripper: open')
            self.robot.GripperOpen()

        elif request.command == "close":
            self.get_logger().info(f'MoveGripper: close')
            self.robot.GripperClose()

        elif request.command == "pos":
            if (request.pos > 30.0) or (request.pos < 0):
                response.error = True
                response.error_message = "MoveGripper pos only accepts a float in range [0, 30.0]"

                self.get_logger().warn(response.error_message)
                return response
            
            self.get_logger().info(f'MoveGripper: move to position {request.pos}')
            self.robot.MoveGripper(request.pos)
            try:
                self.robot.WaitGripperMoveCompletion()
            except Exception as e:
                self.get_logger().warn(f"MoveGripper: WaitGripperMoveCompletion error: {e}")
        else:
            self.get_logger().warn(f"ERROR: Invalid command given to MoveGripper service; only accepts 'open', 'close', or 'pos'.")
            response.error = True
            return response
        
        response.error = False
        return response

    def move_lin_rel_wrf_callback(self, request, response):
        """ MoveLinRelWrf: Move the robot tool linearly relative to its current work reference frame.

        Expects:
            - request.x_offset, request.y_offset, request.z_offset: displacement in mm.
            - request.alpha, request.beta, request.gamma: optional orientation changes in degrees.
                (For 4-axis robots, alpha and beta may be ignored and you can pass a value only for gamma.)
        """
        self.get_logger().info(
            f"MoveLinRelWrf: moving relative to work frame: x={request.x_offset}, y={request.y_offset}, z={request.z_offset}, "
            f"alpha={request.alpha}, beta={request.beta}, gamma={request.gamma}"
        )
        try:
            self.robot.MoveLinRelWrf(
                x=request.x_offset,
                y=request.y_offset,
                z=request.z_offset,
                alpha=request.alpha,
                beta=request.beta,
                gamma=request.gamma
            )
            self.get_logger().debug("MoveLinRelWrf: physical move queued.")
            response.success = True
            response.error_message = ""
        except Exception as e:
            self.get_logger().error(f"MoveLinRelWrf: failed {e}")
            response.success = False
            response.error_message = str(e)
        return response
    
    def move_lin_callback(self, request, response):
        """ MoveLin: Move the robot tool linearly relative to its current work reference frame.
        
        This does not wait for the move to complete.

        Expects:
        - request.x, request.y, request.z: coordinates in mm.
        - request.alpha, request.beta, request.gamma: optional orientation in degrees.
            (For 4-axis robots, alpha and beta may be ignored and you can pass a value only for gamma.)
        """
        self.get_logger().info(
            f"MoveLin: moving linearly to x={request.x}, y={request.y}, z={request.z}, "
            f"alpha={request.alpha}, beta={request.beta}, gamma={request.gamma}"
        )
        try:
            self.robot.MoveLin(
                request.x,
                request.y,
                request.z,
                request.alpha,
                request.beta,
                request.gamma
            )
            response.success = True
            response.error_message = ""

            self.get_logger().debug("MoveLin: physical move queued.")
        except Exception as e:
            self.get_logger().error(f"MoveLin: failed {e}")
            response.success = False
            response.error_message = str(e)
        return response
     
    def move_pose_callback(self, request, response):
        """ MovePose: Move the robot tool to a specified pose.

        Does not wait for the move to complete.

        Request inputs:
            - x, y, z: position in mm.
            - alpha, beta, gamma: orientation in degrees.
        
        Response:
            - success [bool]: True if the move was successful, False otherwise.
            - error_message [string]: Error message if the move failed.
        """
        self.get_logger().info(f"MovePose: moving to x={request.x:.1f}, y={request.y:.1f}, z={request.z:.1f}, "
                               f"alpha={request.alpha:.1f}, beta={request.beta:.1f}, gamma={request.gamma:.1f}")
        try:
            # Call the corresponding mecademicpy command
            self.robot.MovePose(
                x=request.x,
                y=request.y,
                z=request.z,
                alpha=request.alpha,
                beta=request.beta,
                gamma=request.gamma
            )

            response.success = True
            response.error_message = ""
            self.get_logger().debug("MovePose: command queued.")
        except mdr.MecademicException as e: # Catch specific Mecademic errors if possible
            self.get_logger().error(f"MovePose: failed: {e}")
            # Check if it's specifically a singularity at the target or unreachable
            err_code = self.robot.GetStatusRobot().error_status # Get error code if possible
            response.success = False
            response.error_message = f"Error {err_code}: {e}"
        except Exception as e: # Catch any other unexpected errors
            self.get_logger().error(f"MovePose failed with unexpected error: {e}")
            response.success = False
            response.error_message = str(e)
        return response
           
    def get_rt_joint_torq_callback(self, request, response):
        """GetRtJointTorq: Get real-time joint torques.

        Request inputs:
            - include_timestamp: whether to include a timestamp in the response.
            - synchronous_update: whether to update the data synchronously.
            - timeout: timeout in seconds.

        Response:
            - timestamp: timestamp of the data.
            - torques: list of joint torques.
            - success: whether the request was successful.
            - error_message: error message if the request failed.
        """
        self.get_logger().info("GetRtJointTorq: received request")
        try:
            # Call the mecademicpy function to retrieve joint torque data.
            rt_data = self.robot.GetRtJointTorq(include_timestamp=request.include_timestamp,
                                                synchronous_update=request.synchronous_update,
                                                timeout=request.timeout)
            
            # Since the underlying data is stored in self._robot_rt_data.rt_joint_torq,
            # we assume rt_data is a TimestampedData object with attributes 'timestamp' and 'data'.
            if request.include_timestamp:
                response.timestamp = rt_data.timestamp
                response.torques = rt_data.data
            else:
                response.timestamp = 0
                response.torques = rt_data

            response.success = True
            response.error_message = ""
        except Exception as e:
            self.get_logger().error(f"GetRtJointTorq: failed {e}")
            response.success = False
            response.timestamp = 0
            response.torques = []
            response.error_message = str(e)
        return response

    def set_blending_callback(self, request, response):
        """ SetBlending: Set the blending of the robot.
        
        NOTE I would print/log the blending after it has been set via GetBlending (available through the web interface),
        but this is for some reason not available to my knowledge in the mecademicpy library. In the future if we use a
        different communication protocol, this would be a nice feature / way to double check.

        Request inputs:
            - blending [float] from 0 to 100
        
        Response:
            - error [bool]: True if error occurred, False otherwise.
        """

        self.get_logger().info(f'SetBlending: setting blending to {request.blending}')
        if (request.blending > 100) or (request.blending < 0):
            self.get_logger().error("SetBlending: only accepts a float in range [0, 100]")
            response.error = True
            response.error_message = "SetBlending only accepts a float in range [0, 100]"
            return response

        self.robot.SetBlending(request.blending)

        response.error = False
        return response

    def set_gripper_force_callback(self, request, response):
        """ SetGripperForce: Set the force of the gripper.
        
        Request inputs:
            - gripper_force: from 5 to 100, which is a percentage of the maximum force the MEGP 25E gripper can hold (40N).
        
        Response:
            - error [bool]: True if error occurred, False otherwise.
        """
        self.get_logger().info(f'SetGripperForce: setting gripper force to {request.gripper_force}')
        
        if (request.gripper_force > 100) or (request.gripper_force < 5):
            self.get_logger().error("SetGripperForce: only accepts a float in range [5, 100]")
            response.error = True
            response.error_message = "SetGripperForce only accepts a float in range [5, 100]"
            return response

        self.robot.SetGripperForce(request.gripper_force)
        response.error = False
        return response

    def set_gripper_vel_callback(self, request, response):
        """ SetGripperVel: Set the velocity of the gripper.
        
        Request inputs:
            - gripper_vel: from 5 to 100, which is a percentage of the maximum finger velocity of the MEGP 25E gripper (∼100 mm/s).
        
        Response:
            - error [bool]: True if error occurred, False otherwise.
        """
        self.get_logger().info(f'SetGripperVel: setting gripper velocity to {request.gripper_vel}')
        
        if (request.gripper_vel > 100) or (request.gripper_vel < 5):
            self.get_logger().error("SetGripperVel: only accepts a float in range [5, 100]")
            response.error = True
            response.error_message = "SetGripperVel only accepts a float in range [5, 100]"
            return response

        self.robot.SetGripperVel(request.gripper_vel)
        response.error = False
        return response

    def set_joint_vel_callback(self, request, response):
        """ SetJointVel: Set the velocity of the joints.
        
        Request inputs:
            - joint_vel: from 0.001 to 100, which is a percentage of maximum joint velocities.
                - NOTE while you can specify the velocity as .001, I do not recommend it, as it was so slow I did not visually see
                movement. 1 works pretty well for moving slowly; I also do not recommend 100, as that is dangerously fast.
                - NOTE see the meca programming manual (https://cdn.mecademic.com/uploads/docs/meca500-r3-programming-manual-8-3.pdf)
                for more information; the velocities of the different joints are set proportionally as a function
                of their max speed.
        Response:
            - error [bool]: True if error occurred, False otherwise.
        """
        self.get_logger().info(f'SetJointVel: setting joint velocity to {request.joint_vel}')
        
        if (request.joint_vel > 100) or (request.joint_vel < .001):
            self.get_logger().error("SetJointVel: only accepts a float in range [.001, 100]")
            response.error = True
            response.error_message = "SetJointVel only accepts a float in range [.001, 100]"
            return response

        self.robot.SetJointVel(request.joint_vel)
        response.error = False
        return response

    def set_joint_acc_callback(self, request, response):
        """
        Request inputs:
            - joint_acc: from 0.001 to 150, which is a percentage of maximum acceleration of the joints, ranging from 0.001% to 150%
        Response:
            - error [bool]: True if error occurred, False otherwise.
        """
        self.get_logger().info(f'SetJointAcc: setting joint acceleration to {request.joint_acc}')
        
        if (request.joint_acc > 150) or (request.joint_acc < .001):
            self.get_logger().error("SetJointAcc: only accepts a float in range [.001, 100]")
            response.error = True
            response.error_message = "SetJointAcc only accepts a float in range [.001, 100]"
            return response

        self.robot.SetJointAcc(request.joint_acc)
        response.error = False
        return response

    # ActionServer endpoints
    def _execute_joint_trajectory_callback(self, goal_handle):
        """ FollowJointTrajectory: Follow a joint trajectory.

        Request inputs:
            - trajectory: a trajectory to follow.
        """

        traj = goal_handle.request.trajectory
        self.get_logger().info(f"FollowJointTrajectory: received request with {len(traj.points)} waypoints")

        self.robot.SetBlending(100.0)

        result = FollowJointTrajectory.Result()
        def cancel_trajectory():
            goal_handle.canceled()
            result.error_code = FollowJointTrajectory.Result.SUCCESSFUL
            return result
        
        # record the wall‑clock time we begin
        start_time = time.time()

        for i, pt in enumerate(traj.points):
            # compute when to send this waypoint
            send_at = start_time + pt.time_from_start.sec + pt.time_from_start.nanosec * 1e-9

            # busy‑wait until then (very short sleeps so we don’t hog CPU)
            while time.time() < send_at:
                time.sleep(500e-6)
                # abort if the client has canceled
                if goal_handle.is_cancel_requested:
                    return cancel_trajectory()

            # convert radians→degrees and fire off the next joint command
            degs = [math.degrees(x) for x in pt.positions]
            self.get_logger().debug(f"MoveJoints: streaming waypoint {i+1}/{len(traj.points)} → {degs}")
            if i > len(traj.points) - 3:
                self.robot.SetBlending(0.0)
            
            self.robot.MoveJoints(*degs)

            # abort if the client has canceled
            if goal_handle.is_cancel_requested:
                return cancel_trajectory()
        
        # after all waypoints are sent, we’re done
        goal_handle.succeed()
        result.error_code = FollowJointTrajectory.Result.SUCCESSFUL
        return result 
    
    def _execute_cartesian_trajectory_callback(self, goal_handle):
        """ Runs the requested trajectory. Lazy-loads points into the robot. 
        
        Return of this function does not guarantee the robot has stopped moving.

        Executes a duration-based move between waypoints on the Meca500 robot. Future waypoints are loaded into the robot 'command_buffer_time' early.
        Sleeps in 100us increments, while waiting for loading new datapoints.
        Canceling the trajectory will clear the remaining motion queue and stop the robot.
        """

        traj = goal_handle.request.trajectory
        result = FollowCartesianTrajectory.Result()
        self.get_logger().info(f"Cartesian trajectory: received request with {len(traj.points)} waypoints")

        self.robot.SetBlending(100.0)
        self.robot.SetMoveMode(mdr.MxMoveMode.MX_MOVE_MODE_DURATION) # We want to have duration based movements for this trajectories.

        previous_time_from_start = 0
        previous_dt = None # Default Mecademic setting is 3 seconds, but we always want to set it the first time.

        def cancel_trajectory():
            self.robot.SetMoveMode(mdr.MxMoveMode.MX_MOVE_MODE_VELOCITY)
            self.robot.ClearMotion()
            self.robot.ResumeMotion()
            goal_handle.canceled()
            result.error_code = FollowCartesianTrajectory.Result.SUCCESSFUL
            return result
        
        # record the wall‑clock time we begin
        start_time = time.time()
        try:
            for i, pt in enumerate(traj.points):
                # compute when to send this waypoint and the duration between previous waypoint and this waypoint.
                time_from_start = pt.time_from_start.sec + pt.time_from_start.nanosec * 1e-9
                duration =  time_from_start - previous_time_from_start
                if duration <= 0 and i > 0: # We accept zero duration only on the first point.
                    self.get_logger().warn(f"Cartesian trajectory: Skipping waypoint {i} with zero or negative duration {duration:0.3f} ({time_from_start:0.3f} - {previous_time_from_start:0.3f})")
                    continue

                previous_time_from_start = time_from_start
                
                if previous_dt != duration: # Need to update duration per move.
                    self.robot.SetMoveDuration(duration)
                    previous_dt = duration
                
                # convert quaternions to degrees and fire off the next joint command
                alpha_deg, beta_deg, gamma_deg = R \
                    .from_quat([pt.pose.orientation.x, pt.pose.orientation.y,
                                pt.pose.orientation.z, pt.pose.orientation.w]) \
                    .as_euler('xyz', degrees=True)

                # Calculate when to send this command 
                # we sent it default:250ms early, mecademic is synchronous so previous moves will still finish.
                send_time = start_time + time_from_start - duration - self._ROBOT_CONFIG['command_buffer_time']

                # busy‑wait until then (we sleep 1ms so we don't hog CPU, and we do not need to react faster)
                while time.time() < send_time:
                    time.sleep(100e-6)
                    # abort if the client has canceled
                    if goal_handle.is_cancel_requested:
                        return cancel_trajectory()

                self.get_logger().debug(f"Cartesian trajectory: streaming waypoint {i+1}/{len(traj.points)} → {pt.pose.position.x} / {pt.pose.position.y} / {pt.pose.position.z}")
                
                self.robot.MovePose(x=pt.pose.position.x, 
                                    y=pt.pose.position.y, 
                                    z=pt.pose.position.z,
                                    alpha=alpha_deg, 
                                    beta=beta_deg, 
                                    gamma=gamma_deg)
                
                # abort if the client has canceled
                if goal_handle.is_cancel_requested:
                    return cancel_trajectory()
        except Exception as e:
            self.get_logger().error(f"Cartesian trajectory: execution failed: {e}")
            goal_handle.abort()
            result.error_code = FollowCartesianTrajectory.Result.PATH_TOLERANCE_VIOLATED
            return result
        finally:
            try:
                end_time = time.time()
                self.get_logger().info(f"Cartesian trajectory: execution time: {end_time - start_time:0.2f} seconds (Robot can still be moving.)")

                # Reset to velocity mode after trajectory completion
                self.robot.SetMoveMode(mdr.MxMoveMode.MX_MOVE_MODE_VELOCITY)
                # Default Blending is on.
                self.robot.SetBlending(100.0)
            except Exception as e:
                self.get_logger().warn(f"Cartesian trajectory: failed to reset move mode: {e}")


        # All waypoints have been sent. This does not mean that the robot has stopped moving.
        goal_handle.succeed()
        result.error_code = FollowCartesianTrajectory.Result.SUCCESSFUL
        return result 
    

def main(args=None):
    """
    Run this node in the following manner, passing in the name associated with the robot as the namespace (__ns):
    ros2 run meca_controller meca_driver --ros-args -r __ns:=/robot1

    Side note, in the future if you want additional parameters/arguments passed in you can do: -p param_name:=param_value
    and access it inside the node class above by doing:
            self.declare_parameter('param_name', 'default value')
            print(self.get_parameter('param_name').value)
    """
    rclpy.init(args=args)
    
    node = Meca_Driver(robot_config=ROBOT1)
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        print('disconnecting…')
        node.stop()
    except mdr.DisconnectError:
        print('disconnecting due to DisconnectError…')
        node.stop_after_disconnect_error()
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()     

if __name__ == "__main__":
    main()
