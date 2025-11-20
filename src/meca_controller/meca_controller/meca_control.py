#!/usr/bin/env python3

"""
Meca_Control, is used to talk with a Mecademic robot and implement common tasks
necessary to handle pick and place tasks in a nano-fabrication environment.

To implement user-code it is suggested to inherit from this class and implement the run function.

This class heavily benefited from work by Jessica Myers (https://github.com/myersjm/mecademic-ros2, University of Illinois Urbana-Champaign)
"""

import math
import time
from functools import cached_property

import numpy as np

import rclpy # python library for ros2. needed for every ros2 node
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup

import tf2_ros
from scipy.spatial.transform import Rotation as R

from geometry_msgs.msg import Pose
from sensor_msgs.msg import JointState

from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import MotionPlanRequest, WorkspaceParameters, Constraints, JointConstraint

from custom_interfaces.srv import MoveJoints
from custom_interfaces.srv import GoToPose, MoveGripper, SetBlending, SetGripperForce, SetGripperVel, SetJointVel, SetJointAcc, MoveLin, MoveLinRelWrf, MovePoseEuler, WaitIdle, GetRtJointTorq, ClearMotion, Home
from custom_interfaces.msg import RobotStatus, GripperStatus
from custom_interfaces.msg import CartesianTrajectory, CartesianTrajectoryPoint, CartesianTolerance
from custom_interfaces.action import FollowCartesianTrajectory

from meca_controller.meca_settings import ROBOT1 # ADJUST THESE GLOBAL CONSTANTS

# never use scientific notation, always 3 digits after the decimal point
np.set_printoptions(suppress=True, precision=3)

_DEFAULT_SETTINGS = {
    'command_buffer_time': 0.25,  # seconds, For commands that are sent upfront, how big should the time buffer be with respect to execution time.
    'monitoring_interval': 0.05,  # seconds, Set monitoring interval for real time data refresh time
    'move_speed': 40,             # percent, Speed for moving between positions
    'timeout_ros_connections': 5  # seconds 
}

class MecaControl(Node):
    def __init__(self, node_name:str = "meca_control", robot_namespace:str = "", 
                       settings:dict = _DEFAULT_SETTINGS, enable_moveit:bool = True):
        super().__init__(node_name)

        self._robot_namespace = robot_namespace
        self._enable_moveit = enable_moveit
        self._settings = _DEFAULT_SETTINGS | settings

        self._initialization_future = rclpy.task.Future()

        # Subscribe to keep up to date on current joint angles, pose, and status of the robot
        # We always just want the latest information.
        # FIXME would be great if joint states are also inside the namespace
        self._initialize_subscribers()
        self._initialize_action_clients()

        # TF buffer for coordinate transformations
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        self.meca_arm_joint_names = [
            "meca_axis_1_joint", "meca_axis_2_joint", "meca_axis_3_joint",
            "meca_axis_4_joint", "meca_axis_5_joint", "meca_axis_6_joint"
        ]

    def _initialize_subscribers(self):
        """ Initialize subscribers.
        """
        self.get_logger().debug(f"init: Initializing subscribers for {self._robot_namespace}.")

        def update_robot_joints_callback(joints):
            """ JointState (has .position and .velocity fields) """
            self.current_joints_robot = joints 
            if self._check_initialized(): self._initialization_future.set_result(True)

        def update_robot_pose_callback(pose_msg):
            """ Pose (has .position and .orientation fields) """
            self.current_pose_robot = pose_msg
            if self._check_initialized(): self._initialization_future.set_result(True)

        def update_robot_gripper_status_callback(gripper_status):
            self.current_gripper_status_robot = gripper_status # custom msg type
            if self._check_initialized(): self._initialization_future.set_result(True)

        def update_robot_status_callback(robot_status):
            self.current_robot_status = robot_status # custom msg type
            if self._check_initialized(): self._initialization_future.set_result(True)

        self._subscriber = {}
        self._subscriber['joint']   = self.create_subscription(JointState, "/joint_states",
                                                               update_robot_joints_callback, 1)
        self._subscriber['pose']    = self.create_subscription(Pose, f"{self._robot_namespace}/pose",
                                                               update_robot_pose_callback, 1)
        self._subscriber['gripper'] = self.create_subscription(GripperStatus, f"{self._robot_namespace}/gripper_status",
                                                               update_robot_gripper_status_callback, 1)
        self._subscriber['robot']   = self.create_subscription(RobotStatus, f"{self._robot_namespace}/robot_status",
                                                               update_robot_status_callback, 1)

        self.current_joints_robot = None
        self.current_pose_robot = None
        self.current_gripper_status_robot = None
        self.current_robot_status = None

    def _initialize_action_clients(self):
        """ Initialize action clients for MoveIt and Cartesian Trajectory.
        """
        timeout_sec = self._settings['timeout_ros_connections']

        # Initialize MoveIt if enabled
        if self._enable_moveit:
            self.get_logger().debug(f"init: Initializing MoveIt action client for {self._robot_namespace}.")
            
            # Add MoveIt2 interface
            self._moveit_callback_group = ReentrantCallbackGroup()
            
            # Action client for MoveIt2
            self._moveit_action_client = ActionClient(
                self,
                MoveGroup,
                'move_action',
                callback_group=self._moveit_callback_group
            )
            
            # Wait for MoveIt to be available
            if not self._moveit_action_client.wait_for_server(timeout_sec=timeout_sec):
                self.get_logger().warn('init: MoveIt action server not available, proceeding without it.')
                self.moveit_available = False
            else:
                self.get_logger().info('init: MoveIt action server found and ready.')
                self.moveit_available = True

        # Initialize Cartesian Trajectory action client
        self._cart_traj_action_client = ActionClient(self, 
                                                     FollowCartesianTrajectory, 
                                                     f'{self._robot_namespace}/meca_arm_controller/follow_cartesian_trajectory')

        # Wait for Cartesian Trajectory to be available with retry logic
        max_retries = 5
        retry_delay = 2.0  # seconds between retries
        
        self.get_logger().info(f"init: Waiting for Cartesian Trajectory action server with {timeout_sec}s timeout...")
        
        for attempt in range(max_retries):
            if self._cart_traj_action_client.wait_for_server(timeout_sec=timeout_sec):
                self.get_logger().info('init: Cartesian Trajectory action server found and ready.')
                return  # Success, exit the function
            
            if attempt < max_retries - 1:
                self.get_logger().warn(f'init: Cartesian Trajectory action server not available (attempt {attempt + 1}/{max_retries}). Retrying in {retry_delay}s...')
                # Add some debugging info
                self.get_logger().debug(f'init: Action server status: {self._cart_traj_action_client.get_state()}')
                time.sleep(retry_delay)
            else:
                self.get_logger().error('init: Cartesian Trajectory action server not available after all retries. Needed for pathplanning.')
                raise Exception('Cartesian Trajectory action server not available. Needed for pathplanning.')

    def _check_initialized(self):
        return (self._initialization_future.done() or 
                (self.current_joints_robot is not None and 
                 self.current_robot_status is not None and 
                 self.current_gripper_status_robot is not None and
                 self.current_pose_robot is not None)
               )

    def wait_for_initialization(self, timeout_sec=None):
        """ Waits for all information channels from the robots to be initialized.

        Parts of this code rely upon getting information from topics published to by the meca_driver.
        Errors will occur if the main code is executed before these are assigned (e.g. motion plan but not yet received the
        current joint states from the subscriber). This function will wait until the critical variables are set so that all
        the code runs as intended.

        Inputs:
            - timeout_sec: amount of time in seconds that should wait to receive data from both robots before beginning code execution;
                           after timeout period, just waits for data from one robot to begin because assumes you are only using 1 robot.
        """
        self.get_logger().info('Waiting for robots to be available.')

        if timeout_sec is None:
            timeout_sec = self._settings['timeout_ros_connections']

        rclpy.spin_until_future_complete(self, self._initialization_future, timeout_sec=timeout_sec)
    
    def run(self):
        """ A space to write user code to control the robots. This should be implemented by a derived class. 
        """
        raise NotImplementedError('Needs to be implemented by a subclass.')
    
    # ------ Clients ----
    @cached_property
    def _client_move_joints(self):
        client = self.create_client(MoveJoints, f'{self._robot_namespace}/move_joints')
        while not client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn("move_joints service not available. Waiting")
        return client

    def move_joints(self, desired_joint_angles, error_tolerance=.1, timeout_length=60):
        """ Move joints to desired angles.

        Inputs:
        - desired_joint_angles (degrees): list-like or np.array of len 6.
        - error_tolerance: (degrees) the joint angle tolerance within which the robot should reach before executing the next command
        - timeout_length: the max amount of time in seconds to wait on the robot to reach the desired position before returning failure

        Returns:
        - is_reached: boolean, whether the robot successfully reached the desired position.
        """
        desired_joint_angles = np.asarray(desired_joint_angles, dtype=np.float64)

        client = self._client_move_joints
        request = MoveJoints.Request()  
        request.requested_joint_angles = desired_joint_angles.tolist()
        future = client.call_async(request)
        rclpy.spin_until_future_complete(self, future) # note that we don't really care about the response (future.result()) itself.

        # 5) Wait for position to be reached, within a tolerance, and under the timeout constraints (to prevent infinite while loops):
        start_time = time.time()
        while rclpy.ok() and (time.time() - start_time) < timeout_length:
            rclpy.spin_once(self) # spin once while we wait to avoid blocking any callbacks. (example: joint angle updates)
            
            # Break out when reaches desired joint angles:
            if self.has_reached_config(desired_joint_angles,
                                       self.current_joints_robot.position,
                                       error_tolerance):
                return True # has reached the desired position
            
            time.sleep(1e-4)
        
        return False # timeout reached or rclpy was interrupted.
    
    @cached_property
    def _client_move_gripper(self):
        client = self.create_client(MoveGripper, f'{self._robot_namespace}/move_gripper')
        while not client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn("MoveGripper service not available, Waiting")
        return client

    def move_gripper(self, position, command="pos"):
        """
        Service call

        Function inputs:
            - namespace (robot for which you want to control)
            - position (see pos below)
            - command: default "pos" (see command below); convenience functions open_gripper and close_gripper will use this
                    variable for making the service call.

        Request inputs:
            - command [String]: {"open", "close", "pos"}
            - pos [float]: gripper position in mm in range [0, 5.6]
        
        Response:
            - error [bool]: True if error occurred, False otherwise.
        """
        self.get_logger().info(f"MoveGripper: Request {command} / {position}mm.")
        client = self._client_move_gripper
        
        request = MoveGripper.Request()
        request.command = command
        request.pos = float(position)

        future = client.call_async(request)
        rclpy.spin_until_future_complete(self, future) 
        response = future.result()
        if response.error:
            raise Exception(f"MoveGripper: Request failed. See driver for specific error message.")

        self.get_logger().debug(f"MoveGripper: Request completed.")
        return True
    
    def open_gripper(self):
        self.move_gripper(0, command="open")

    def close_gripper(self):
        self.move_gripper(0, command="close")
    
    @cached_property
    def _client_move_lin_rel_wrf(self):
        client = self.create_client(MoveLinRelWrf, f'{self._robot_namespace}/move_lin_rel_wrf')
        while not client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn("MoveLinRelWrf: Service not available, waiting.")
        return client

    def move_lin_rel_wrf(self, x, y, z, alpha, beta, gamma):
        client = self._client_move_lin_rel_wrf

        request = MoveLinRelWrf.Request()
        request.x_offset = x
        request.y_offset = y
        request.z_offset = z
        request.alpha = alpha
        request.beta = beta
        request.gamma = gamma

        self.get_logger().info(f"MoveLinRelWrf: Request {request.x_offset} / {request.y_offset} / {request.z_offset} / {request.alpha} / {request.beta} / {request.gamma}.")
        future = client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        if future.result() is not None:
            if hasattr(future.result(), 'success'):
                if future.result().success:
                    self.get_logger().debug("MoveLinRelWrf: Request completed.")
                    return True
                else:
                    self.get_logger().error("MoveLinRelWrf: Request failed.")
                    return False
            else:
                self.get_logger().debug("MoveLinRelWrf: Request sent (no response flag).")
                return False
        else:
            self.get_logger().error("MoveLinRelWrf: No response received.")
            return False
    
    @cached_property
    def _client_move_lin(self):    
        client = self.create_client(MoveLin, f'{self._robot_namespace}/move_lin')
        while not client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn("MoveLin: Service not available, Waiting")
        return client

    def move_lin(self, x, y, z, alpha, beta, gamma):
        client = self._client_move_lin

        request = MoveLin.Request()
        request.x = float(x)
        request.y = float(y)
        request.z = float(z)
        request.alpha = float(alpha)
        request.beta = float(beta)
        request.gamma = float(gamma)
        
        self.get_logger().info(f"MoveLin: Request {request.x} / {request.y} / {request.z} / {request.alpha} / {request.beta} / {request.gamma}.")
        try:
            future = client.call_async(request)
            rclpy.spin_until_future_complete(self, future, timeout_sec=70.0) # Add timeout (slightly > driver WaitIdle)
            
            result = future.result()

            if result is not None:
                # Check if the service definition HAS a success field
                if hasattr(result, 'success'): 
                    if result.success:
                        self.get_logger().debug("MoveLin: Request completed.")
                        return True # Explicit success
                    else:
                        err_msg = getattr(result, 'error_message', 'No error message provided.')
                        self.get_logger().error(f"MoveLin service reported FAILURE: {err_msg}")
                        return False # Explicit failure
                else: # No success field in .srv definition
                    self.get_logger().debug("MoveLin: Request sent (no response flag).")
                    return True # Assume success if call completed without exception
            else:
                # This happens if spin_until_future_complete times out or service died
                self.get_logger().error("MoveLin: No response received.")
                return False 

        except Exception as e:
            self.get_logger().error(f"MoveLin: Exception during request: {e}")
            import traceback
            traceback.print_exc()
            return False

    @cached_property
    def _client_move_pose(self):    
        client = self.create_client(MovePoseEuler, f'{self._robot_namespace}/move_pose')
        while not client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn("MovePose: Service not available, Waiting")
        return client
    
    def move_pose(self, x, y, z, alpha, beta, gamma):
        """Sends a MovePose command without waiting for completion.

        Inputs:
            - x, y, z, alpha, beta, gamma: pose to move to

        Returns:
            - True if the request was sent successfully, False otherwise.
        """
        client = self._client_move_pose
        request = MovePoseEuler.Request()
        request.x = float(x)
        request.y = float(y)
        request.z = float(z)
        request.alpha = float(alpha)
        request.beta = float(beta)
        request.gamma = float(gamma)

        self.get_logger().info(f"MovePose: Request {request.x} / {request.y} / {request.z} / {request.alpha} / {request.beta} / {request.gamma}.")
        future = client.call_async(request)
        try:
            rclpy.spin_until_future_complete(self, future, timeout_sec=30.0)
            result = future.result()
            
            if result is not None:
                if hasattr(result, 'success'):
                    if result.success:
                        self.get_logger().debug("MovePose: Request completed.")
                        return True
                    else:
                        err_msg = getattr(result, 'error_message', 'No error message provided.')
                        self.get_logger().error(f"MovePose service reported FAILURE: {err_msg}")
                        return False
                else:
                    self.get_logger().debug("MovePose: Request sent (no response flag).")
                    return True
            else:
                self.get_logger().error("MovePose: No response received.")
                return False
                
        except Exception as e:
            self.get_logger().error(f"MovePose: Exception during request: {e}")
            return False

    @cached_property
    def _client_get_rt_joint_torq(self):    
        client = self.create_client(GetRtJointTorq, f'{self._robot_namespace}/get_rt_joint_torq')
        while not client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn("GetRtJointTorq: Service not available, waiting")
        return client

    def get_rt_joint_torq(self, include_timestamp, synchronous_update, timeout):
        self.get_logger().info(f"GetRtJointTorq: Request {include_timestamp} / {synchronous_update} / {timeout}s.")
        client = self._client_get_rt_joint_torq

        request = GetRtJointTorq.Request()
        request.include_timestamp = include_timestamp
        request.synchronous_update = synchronous_update
        request.timeout = timeout
        future = client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        if future.result() is not None and future.result().success:
            self.get_logger().info(f"GetRtJointTorq: joint torques: {future.result().torques}")
            return future.result()
        else:
            self.get_logger().error("GetRtJointTorq: Failed to get joint torques")
            return None
    
    @cached_property
    def _client_set_blending(self):    
        client = self.create_client(SetBlending, f'{self._robot_namespace}/set_blending')
        while not client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn("SetBlending: Service not available, waiting")
        return client
        
    def set_blending(self, blending):
        """ Set Blending

        Inputs:
            - blending [float] from 0 to 100

        Returns:
            - True if the request was sent successfully, False otherwise.
        """
        self.get_logger().info(f"SetBlending: Request {blending}.")
        client = self._client_set_blending

        request = SetBlending.Request()
        request.blending = float(blending)

        try:
            future = client.call_async(request)
            rclpy.spin_until_future_complete(self, future, timeout_sec=5.0) # Add timeout

            result = future.result()

            if result is not None:
                # Check if the service definition HAS an error field
                if hasattr(result, 'error'):
                    if not result.error:
                        self.get_logger().info(f"SetBlending: completed for {self._robot_namespace}.")
                        return True # Explicit success
                    else:
                        self.get_logger().error(f"SetBlending: service reported an error (value likely out of range: {blending}).")
                        return False # Explicit failure reported by service
                else: # No error field in .srv definition
                    self.get_logger().debug(f"SetBlending: Request sent (no error field in response, assuming success).")
                    return True # Assume success if call completed
            else:
                self.get_logger().error("SetBlending: No response received.")
                return False

        except Exception as e:
            self.get_logger().error(f"SetBlending: Exception during request: {e}")
            import traceback
            traceback.print_exc()
            return False

    @cached_property
    def _client_set_gripper_force(self):    
        client = self.create_client(SetGripperForce, f'{self._robot_namespace}/set_gripper_force')
        while not client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn("SetGripperForce: Service not available, waiting")
        return client
        
    def set_gripper_force(self, gripper_force: float):
        """ Set Force of Gripper

        Inputs:
            - gripper_force: from 5 to 100, which is a percentage of the maximum force the MEGP 25E gripper can hold (40N).

        Returns:
            - True if the request was sent successfully, False otherwise.
        """
        self.get_logger().info(f"SetGripperForce: Request {gripper_force}.")
        client = self._client_set_gripper_force
        
        request = SetGripperForce.Request()  
        request.gripper_force = float(gripper_force)
        future = client.call_async(request)
        rclpy.spin_until_future_complete(self, future)

        response = future.result()
        if response.error:
            raise Exception(f"SetGripperForce: only accepts a float in range [5, 100]. Requested {gripper_force}.")
        self.get_logger().debug(f"SetGripperForce: Request completed.")
    
    @cached_property
    def _client_set_gripper_vel(self):    
        client = self.create_client(SetGripperVel, f'{self._robot_namespace}/set_gripper_vel')
        while not client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn("SetGripperVel: Service not available, waiting")
        return client
    
    def set_gripper_vel(self, gripper_vel: float):
        """ Set Gripper Velocity

        Inputs:
            - gripper_vel: from 5 to 100, which is a percentage of the maximum finger velocity of the MEGP 25E gripper (∼100 mm/s).

        Returns:
            - True if the request was sent successfully, False otherwise.
        """
        self.get_logger().info(f"SetGripperVel: Request {gripper_vel}.")
        client = self._client_set_gripper_vel
        
        request = SetGripperVel.Request()  
        request.gripper_vel = float(gripper_vel)
        future = client.call_async(request)
        rclpy.spin_until_future_complete(self, future)

        response = future.result()
        if response.error:
            raise Exception(f"SetGripperVel: only accepts a float in range [5, 100]. Requested {gripper_vel}.")
        self.get_logger().debug(f"SetGripperVel: Request completed.")
    
    @cached_property
    def _client_set_joint_vel(self):    
        client = self.create_client(SetJointVel, f'{self._robot_namespace}/set_joint_vel')
        while not client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn("SetJointVel: Service not available, waiting")
        return client
    
    def set_joint_vel(self, joint_vel: float):
        """ Set Joint Velocity

        Inputs:
            - joint_vel: from 0.001 to 100, which is a percentage of maximum joint velocities.
                - NOTE while you can specify the velocity as .001, I do not recommend it, as it was so slow I did not visually see
                movement. 1 works pretty well for moving slowly; I also do not recommend 100, as that is dangerously fast.
                - NOTE see the meca programming manual (https://cdn.mecademic.com/uploads/docs/meca500-r3-programming-manual-8-3.pdf)
                for more information; the velocities of the different joints are set proportionally as a function
                of their max speed.
        Response:
            - error [bool]: True if error occurred, False otherwise.
        """
        self.get_logger().info(f"SetJointVel: Request {joint_vel}.")
        client = self._client_set_joint_vel

        request = SetJointVel.Request()  
        request.joint_vel = float(joint_vel)

        future = client.call_async(request)        
        rclpy.spin_until_future_complete(self, future)

        response = future.result()
        if response.error:
            raise Exception(f"SetJointVel: only accepts a float in range [.001, 100]. Requested {joint_vel}.")
        self.get_logger().debug(f"SetJointVel: Request completed.")

    @cached_property
    def _client_set_joint_acc(self):    
        client = self.create_client(SetJointAcc, f'{self._robot_namespace}/set_joint_acc')
        while not client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn("SetJointAcc: Service not available, waiting")
        return client
        
    def set_joint_acc(self, joint_acc):
        """ Set Joint Acceleration

        Inputs:
            - joint_acc: from 0.001 to 150, which is a percentage of maximum acceleration of the joints, ranging from 0.001% to 150%
        Response:
            - error [bool]: True if error occurred, False otherwise.
        """
        self.get_logger().info(f"SetJointAcc: Request {joint_acc}.")
        client = self._client_set_joint_acc
        
        request = SetJointAcc.Request()  
        request.joint_acc = float(joint_acc)

        future = client.call_async(request)
        rclpy.spin_until_future_complete(self, future)

        response = future.result()
        if response.error:
            raise Exception(f"SetJointAcc: only accepts a float in range [.001, 150]. Requested {joint_acc}.")
        self.get_logger().debug(f"SetJointAcc: Request completed.")

    @cached_property
    def _client_wait_idle(self):    
        client = self.create_client(WaitIdle, f'{self._robot_namespace}/wait_idle')
        while not client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn("WaitIdle: Service not available, waiting")
        return client

    def wait_idle(self, timeout_s: float = 60.0) -> bool:
        """Wait for the robot to be idle.

        Inputs:
            - timeout_s: timeout in seconds for the wait idle service

        Response:
            - error [bool]: True if error occurred, False otherwise.
        """
        self.get_logger().info(f"WaitIdle: Timeout {timeout_s}s, controller will wait +5s for service response.")
        request = WaitIdle.Request()
        request.timeout_sec = float(timeout_s) # This is the timeout for the robot's internal WaitIdle
        
        future = self._client_wait_idle.call_async(request)
        
        # Spin until the service call itself completes (or this controller-side timeout is hit)
        # The service call will block on the driver side until the robot is idle or driver's timeout.
        rclpy.spin_until_future_complete(self, future, timeout_sec = timeout_s + 5.0) # Controller timeout
        
        result = future.result()
        if result is not None:
            if result.success:
                self.get_logger().debug("WaitIdle: Robot is idle.")
                return True
            else:
                self.get_logger().error(f"WaitIdle: Service reported failure: {result.error_message}")
                return False
        else:
            self.get_logger().error("WaitIdle: No response from service (controller timed out waiting for service response).")
            return False

    @cached_property
    def _client_clear_motion(self):    
        client = self.create_client(ClearMotion, f'{self._robot_namespace}/clear_motion')
        while not client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn("ClearMotion: Service not available, waiting")
        return client
    
    def clear_motion(self, timeout_s: float = 10.0) -> bool:
        """ Clear the robot's motion buffer.

        Inputs:
            - timeout_s: timeout in seconds for the clear motion service

        Response:
            - error [bool]: True if error occurred, False otherwise.
        """       
        self.get_logger().info(f"ClearMotion: Requested, with timeout {timeout_s}s")
        
        request = ClearMotion.Request()
        future = self._client_clear_motion.call_async(request)
        
        rclpy.spin_until_future_complete(self, future, timeout_sec=timeout_s)
        
        result = future.result()
        if result is not None:
            if result.success:
                self.get_logger().debug(f"ClearMotion: Service reported success.")            
                return True
            else:
                self.get_logger().error(f"ClearMotion: Service reported failure: {result.message}")    
                return False
        else:
            self.get_logger().error("ClearMotion: No response from service (controller timed out).")
            return False

    @cached_property
    def _client_home(self):    
        client = self.create_client(Home, f'{self._robot_namespace}/home')
        while not client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn("Home: Service not available, waiting")
        return client
    
    def home(self, timeout_s: float = 60.0) -> bool:
        """Calls the Home service to home the robot.

        Inputs:
            - timeout_s: timeout in seconds for the home service

        Response:
            - error [bool]: True if error occurred, False otherwise.
        """
        self.get_logger().info(f"Home: Requested, with timeout {timeout_s}s")
        
        request = Home.Request()
        future = self._client_home.call_async(request)
        
        rclpy.spin_until_future_complete(self, future, timeout_sec=timeout_s)
        
        result = future.result()
        if result is not None:
            if result.success:
                self.get_logger().debug(f"Home: Service reported success.")
                return True
            else:
                self.get_logger().error(f"Home: Service reported failure: {result.error_message}")
                return False
        else:
            self.get_logger().error("Home: No response from service (controller timed out).")
            return False
    
    # ------

    def verify_position(self, x: float, y: float, z: float, a: float, b: float, g: float, 
                        pos_tol_mm: float = 0.5, orient_tol_deg: float = 0.5, vel_threshold: float = 0.1):
        """ Verify position is close enough to target (x,y,z, a, b, g) in mm and degree.

        Inputs:
            - x: target x position in mm
            - y: target y position in mm
            - z: target z position in mm
            - a: target a orientation in degrees
            - b: target b orientation in degrees
            - g: target g orientation in degrees
            - pos_tol_mm: position tolerance in mm
            - orient_tol_deg: orientation tolerance in degrees
            - vel_threshold: velocity threshold in mm/s (FIXME: currently ignored)

        Returns:
            - True if the position is close enough to the target, False otherwise.
        """
        self.spin_until_timer(0.25) # Make sure remaining callbacks could be run to get latest position.

        current_p = self.current_pose_robot
        curr_pos = np.array([current_p.position.x, current_p.position.y, current_p.position.z])
        curr_orient_deg = np.array([current_p.orientation.x, current_p.orientation.y, current_p.orientation.z])
        
        target_pos = np.array([x, y, z])
        target_orient_deg = np.array([a, b, g])

        pos_error = np.linalg.norm(curr_pos - target_pos)
        
        # Convert current and target orientations to quaternions
        curr_quat = R.from_euler('xyz', curr_orient_deg, degrees=True).as_quat()
        target_quat = R.from_euler('xyz', target_orient_deg, degrees=True).as_quat()

        # Quaternion distance (angle between orientations)
        # Note: quaternions q and -q represent the same orientation
        # So we take the minimum of the two possible distances
        dot_product = np.abs(np.dot(curr_quat, target_quat))
        # Clamp to [-1, 1] to avoid numerical errors
        dot_product = np.clip(dot_product, -1.0, 1.0)
        # Convert to angle in degrees
        orient_error = 2.0 * np.arccos(dot_product) * 180.0 / np.pi
        
        # Velocity check
        is_still = True
        if self.current_joints_robot and self.current_joints_robot.velocity:
            joint_vels = np.abs(self.current_joints_robot.velocity[:6])
            # FIXME: this currently does not work because of velocity data somehow being wrong.
        #     is_still = np.all(joint_vels < vel_threshold)

        if pos_error <= pos_tol_mm and orient_error <= orient_tol_deg and is_still:
            return True
        else:
            self.get_logger().info(f"Position error: {pos_error} mm, Orientation error: {orient_error} deg, Velocity: {joint_vels}")
            return False
    
    def verify_joints(self, target_joints_deg: np.ndarray, 
                      tol_deg: float = 0.1, vel_threshold: float = 0.1) -> bool:
        """ Verify joint angles are close enough to target (in degrees).
        
        Inputs:
            - target_joints_deg: target joint angles in degrees
            - tol_deg: tolerance in degrees
            - vel_threshold: velocity threshold in degrees/s

        Returns:
            - True if the joint angles are close enough to the target, False otherwise.
        """
        self.spin_until_timer(0.25) # Make sure remaining callbacks could be run to get latest position.

        current_joints_deg = np.degrees(np.array(self.current_joints_robot.position[:6]))
        joint_vels = np.abs(np.array(self.current_joints_robot.velocity[:6]))

        if np.all(np.abs(current_joints_deg - target_joints_deg) <= tol_deg) and np.all(joint_vels < vel_threshold):
            return True
        else:
            self.get_logger().info(f"Joint error: {current_joints_deg - target_joints_deg}, Velocity: {joint_vels}")
            return False

    def has_reached_config(self, desired_joint_angles, current_joint_angles, error_tolerance=.1):
        """ Check if the robot has reached its desired configuration, to a tolerance (in degrees).

        Inputs:
            - desired_joint_angles: desired joint angles in degrees
            - current_joint_angles: current joint angles in degrees
            - error_tolerance: error tolerance in degrees

        Returns:
            - True if the robot has reached its desired configuration, False otherwise. 
        """
        desired_radians = np.radians(desired_joint_angles)
        #FIXME: This could be using all from numpy.
        return np.all([self.within_tolerance(desired_rad, curr_joint, error_tolerance) 
                    for (desired_rad, curr_joint) in zip(desired_radians, current_joint_angles)])
    
    # t - .0001 <= x <= t + .0001:
    def within_tolerance(self, desired_joint_pos, current_joint_pos, error_tolerance):
        return (((desired_joint_pos - error_tolerance) <= current_joint_pos) and 
                (current_joint_pos <= (desired_joint_pos + error_tolerance)))
    
    def create_goal_constraints(self, target_joints_rad):
        """ Create goal constraints based on target joint angles.

        This function constructs a Constraints message containing JointConstraint entries
        for each joint in the planning group ("meca_arm"). It assumes your robot has six joints.
        
        Inputs:
            - target_joints_rad: NumPy array of target joint positions (in radians).
        
        Returns:
            - constraints: A list containing one Constraints message.
        """
        
        constraints = Constraints()
        # List the joint names as defined in your SRDF for the arm.
        joint_names = [
            "meca_axis_1_joint",
            "meca_axis_2_joint",
            "meca_axis_3_joint",
            "meca_axis_4_joint",
            "meca_axis_5_joint",
            "meca_axis_6_joint"
        ]
        # Create a constraint for each joint.
        for i, name in enumerate(joint_names):
            jc = JointConstraint()
            jc.joint_name = name
            jc.position = float(target_joints_rad[i])
            jc.tolerance_above = 0.01  # 0.01 rad tolerance (adjust as needed)
            jc.tolerance_below = 0.01
            jc.weight = 1.0
            constraints.joint_constraints.append(jc)

        # Return the constraints as a list (MoveIt expects a list of constraints).
        return [constraints]

    def plan_to_joint_target(self, joint_angles: np.ndarray):
        """ Plans a trajectory to reach the desired joint angles using MoveIt's planning action.

        Inputs:
            - joint_angles: desired joint angles in degrees

        Returns:
            - success: bool: True if the planning was successful, False otherwise.
            - trajectory: The planned trajectory.
        """
        if not self.moveit_available:
            self.get_logger().error('PlanToJointTarget: MoveIt is not available.')
            return False, None

        # Convert target joint angles from degrees to radians.
        # (MoveIt expects joint angles in radians.)
        target_joints_rad = np.radians(joint_angles)

        # Create the goal message for the MoveGroup action.
        goal_msg = MoveGroup.Goal()

        # Create and fill in a MotionPlanRequest.
        motion_request = MotionPlanRequest()
        
        # Define workspace parameters. Use your robot's base frame.
        workspace = WorkspaceParameters()
        # Update the frame to match your robot. According to your SRDF,
        workspace.header.frame_id = "meca_base_link"
        workspace.header.stamp = self.get_clock().now().to_msg()
        motion_request.workspace_parameters = workspace

        # Use the current state as the start state.
        # Setting is_diff = True tells MoveIt to use the current state.
        motion_request.start_state.is_diff = True

        # Set goal constraints.
        # Use a helper function to create joint constraints based on the target joint values.
        motion_request.goal_constraints = self.create_goal_constraints(target_joints_rad)

        # Set additional planning parameters (adjust as needed for your robot)
        motion_request.max_velocity_scaling_factor = 0.8
        motion_request.max_acceleration_scaling_factor = 0.8
        motion_request.allowed_planning_time = 5.0
        motion_request.num_planning_attempts = 10

        # Set planner_id to "RRTConnect", a popular sampling-based planner.
        motion_request.planner_id = "RRTConnect"

        # IMPORTANT: Set the planning group name.
        # You find this in your SRDF.
        motion_request.group_name = "meca_arm"
        
        # Assign the planning request to the goal message.
        goal_msg.request = motion_request

        self.get_logger().info('Sending planning request to MoveIt...')
        send_goal_future = self._moveit_action_client.send_goal_async(goal_msg)
        rclpy.spin_until_future_complete(self, send_goal_future)
        goal_handle = send_goal_future.result()

        if not goal_handle.accepted:
            self.get_logger().error('MoveIt planning request was rejected.')
            return False, None

        self.get_logger().info('Planning request accepted; waiting for result...')
        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        result = result_future.result().result  # Extract the result from the action result.

        # Check if the planning was successful; typically error_code.val == 1 means success.
        if result.error_code.val != 1:
            self.get_logger().error(f'Planning failed with error code: {result.error_code.val}')
            return False, None

        self.get_logger().info('Successfully planned a trajectory.')
        return True, result.planned_trajectory
    
    def wait_for_motion_complete(
        self,
        target_joints_deg: np.ndarray,
        tol_deg: float = 0.5,
        vel_threshold: float = 0.01,
        timeout_s: float = 60.0
    ) -> bool:
        """ Wait for the robot to reach the target joint angles and have joint velocities below vel_threshold.
        
        Blocks until the robot has reached `target_joints_deg` (deg)
        AND joint velocities are below vel_threshold (rad/s),
        or until timeout_s elapses.  Then sends one final MoveJoints
        service call and waits for it to complete.
        
        Inputs:
            - target_joints_deg: target joint angles in degrees
            - tol_deg: tolerance in degrees
            - vel_threshold: velocity threshold in degrees/s
            - timeout_s: timeout in seconds

        Returns:
            - True if the robot has reached the target joint angles and has joint velocities below vel_threshold, False otherwise.
        """
        start = self.get_clock().now()

        request = WaitIdle.Request()
        request.timeout_sec = float(timeout_s) # This is the timeout for the robot's internal WaitIdle
        
        future = self._client_wait_idle.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec = timeout_s - 0.5)
        result = future.result()
        if result is not None:
            if not result.success:
                self.get_logger().error(f"WaitIdle: Service reported failure: {result.error_message}")
                return False
        else:
            self.get_logger().error("WaitIdle: No response from service.")
            return False
        
        # Final nudge to move joints in perfect position.
        client = self._client_move_joints
        req = MoveJoints.Request()
        req.requested_joint_angles = target_joints_deg.tolist()
        fut = client.call_async(req)
        rclpy.spin_until_future_complete(self, fut, timeout_sec=timeout_s - (time.time() - start))
        if fut.result() is None:
            self.get_logger().error("Final nudge MoveJoints call failed or timed out")
            return False

        return self.verify_joints(target_joints_deg, tol_deg, vel_threshold)

    def move_circular_time_freq(self, radius_mm: float, duration_seconds: float, start_pose: Pose, 
                                num_points: int = 16, frequency: float = 1.0, clockwise: bool = True, 
                                move_z_distance_mm: float = 20.0, move_z_time_seconds: float = 0.2,
                                block: bool = True):
        """ Moves the robot TCP in circles for a specified time duration with a given frequency.
        
        It allows for a z-displacement at beginning and end.
        
        This method generates a complete Cartesian trajectory with timing information and submits
        it to the robot controller via the FollowCartesianTrajectory action client. 
        The trajectory includes all waypoints that fit within the specified duration at the given frequency.
        
        Inputs:
            - radius_mm: The radius of the circle in mm.
            - duration_seconds: Total duration for circular motion in seconds.
            - start_pose: Pose to start the circular motion from. (x,y,z,a,b,g)
            - num_points: Number of waypoints to approximate each complete circle lap.
            - frequency: Number of laps per second (must be between 0.1 and 3.0).
            - clockwise: Direction of the circle motion. True for clockwise, False for counter-clockwise.
            - move_z_distance_mm: Distance to move up/down before/after completing circular motion. Default 20.0mm.
            - move_z_time_seconds: Time to cover the distance. Default 0.2 s
            - block: Whether to block until the circular motion is complete. Default True.
        Returns:
            - success: bool: True if the circular motion completed successfully, False otherwise.
        """
        self.get_logger().info(f"move_circular_time_freq: Starting timed circular move: Radius={radius_mm} mm, Points={num_points}," +
                               f"Duration={duration_seconds}s, Frequency={frequency} Hz," + 
                               f"Start Pose={start_pose.position.x:.1f}, {start_pose.position.y:.1f}, {start_pose.position.z:.1f}," +
                               f"{start_pose.orientation.x:.1f}, {start_pose.orientation.y:.1f}, {start_pose.orientation.z:.1f}")

        if not (0.1 <= frequency <= 3.0):
            self.get_logger().error(f"move_circular_time_freq: Frequency must be between 0.1 and 3.0, got {frequency}")
            return False
        
        if duration_seconds <= 0.5:
            self.get_logger().error("move_circular_time_freq: Duration must be greater than 0.5 seconds.")
            return False

        if num_points < 3:
            self.get_logger().error("move_circular_time_freq: Number of points for circle must be at least 3.")
            return False
           
        # Store the initial pose values (center and orientation)
        # Need to move down from current pose first.
        cx = start_pose.position.x
        cy = start_pose.position.y
        cz_raised = start_pose.position.z
        cz = cz_raised - move_z_distance_mm
        ca = start_pose.orientation.x 
        cb = start_pose.orientation.y
        cg = start_pose.orientation.z 
        
        self.get_logger().info(f"move_circular_time_freq: Circle center based on current pose: "
                               f"X={cx:.1f}, Y={cy:.1f}, Z={cz:.1f} / (-{move_z_distance_mm:.1f}), "
                               f"A={ca:.1f}, B={cb:.1f}, G={cg:.1f}")
            
        # Calculate total number of waypoints based on frequency
        circle_waypoints = int(frequency * duration_seconds * num_points)
        if circle_waypoints < num_points:
            self.get_logger().warning(
                f"move_circular_time_freq: Frequency {frequency} Hz and duration {duration_seconds}s only allow {circle_waypoints} waypoints, "
                f"but {num_points} points per circle requested. The circle will not be complete, which is expected for this duration."
            )

        self.get_logger().debug(f"move_circular_time_freq: Generating {circle_waypoints} waypoints for {duration_seconds}s duration at {frequency} Hz frequency")
        
        # --- Generate all waypoints that fit into the time duration ---
        waypoints = []
        angle_step = 2 * math.pi / num_points
        
        for i in range(circle_waypoints): 
            point_in_circle = i % num_points + 1
            
            theta = point_in_circle * angle_step
            if clockwise:
                theta = -theta 

            target_x = cx + radius_mm * math.cos(theta)
            target_y = cy + radius_mm * math.sin(theta)
            target_z = cz 
            target_alpha = ca
            target_beta = cb
            target_gamma = cg
            
            waypoints.append((target_x, target_y, target_z, target_alpha, target_beta, target_gamma))

        self.get_logger().debug(f"move_circular_time_freq: Request Cartesian trajectory with {len(waypoints)} waypoints.")
        
        goal = FollowCartesianTrajectory.Goal()

        trajectory = CartesianTrajectory()
        trajectory.header.frame_id = "meca_base_link"
        trajectory.header.stamp = self.get_clock().now().to_msg()
        trajectory.controlled_frame = "meca_tcp_link"
        
        time_start_circle = 0
        if move_z_distance_mm > 0: # need to start with moving down.
            # Give some time in trajectory to move to first waypoint
            time_start_circle = move_z_time_seconds 

        time_step_seconds = duration_seconds / circle_waypoints

        for i, wp in enumerate(waypoints):
            x, y, z, a, b, g = wp
            
            point = CartesianTrajectoryPoint()
            
            # Set timing (time from start of trajectory)
            time_from_start = time_start_circle + i * time_step_seconds
            point.time_from_start.sec = int(time_from_start)
            point.time_from_start.nanosec = int((time_from_start - int(time_from_start)) * 1e9)
            
            # Create pose (convert from Euler angles to quaternion)            
            # Convert Euler angles (degrees) to quaternion
            # NOTE: This assumes the angles are in the order A, B, G (roll, pitch, yaw)
            quat = R.from_euler('xyz', [a, b, g], degrees=True).as_quat()           
            pose = self.create_pose(x, y, z, *quat)
            
            point.pose = pose 
            trajectory.points.append(point)
        
        goal.trajectory = trajectory
        
        # NOTE:Tolerances not used.
        goal.path_tolerance = CartesianTolerance()       
        goal.goal_tolerance = CartesianTolerance()
        
        # NOTE: Goal time tolerance not used.
        goal.goal_time_tolerance.sec = 1
        goal.goal_time_tolerance.nanosec = 0
        
        if move_z_distance_mm > 0:
            # Return to lifted position
            point = CartesianTrajectoryPoint()

            time_from_start = time_start_circle + (len(waypoints)-1) * time_step_seconds + move_z_time_seconds
            point.time_from_start.sec = int(time_from_start)
            point.time_from_start.nanosec = int((time_from_start - int(time_from_start)) * 1e9)

            quat = R.from_euler('xyz', [ca, cb, cg], degrees=True).as_quat()
            pose = self.create_pose(cx, cy, cz_raised, *quat)
            point.pose = pose
            trajectory.points.append(point)
        
        self.get_logger().debug(f"move_circular_time_freq: Submitting Cartesian trajectory with {len(trajectory.points)} total points / ({len(waypoints)} circle waypoints)")
        start_time = time.time()

        send_goal_future = self._cart_traj_action_client.send_goal_async(goal)

        success = False
        # Wait for goal to be accepted
        rclpy.spin_until_future_complete(self, send_goal_future, timeout_sec=10.0)
        if not send_goal_future.done():
            self.get_logger().error("move_circular_time_freq: Failed to send trajectory goal within timeout")
            success = False
        else:
            goal_handle = send_goal_future.result()
            if not goal_handle.accepted:
                self.get_logger().error("move_circular_time_freq: Trajectory goal was rejected")
                success = False
            else:
                self.get_logger().debug("move_circular_time_freq: Trajectory goal accepted, waiting for completion...")
                
                # Wait for result
                get_result_future = goal_handle.get_result_async()
                rclpy.spin_until_future_complete(self, get_result_future, timeout_sec=duration_seconds + 10.0)
                
                if not get_result_future.done():
                    self.get_logger().error("move_circular_time_freq: Trajectory execution timed out")
                    success = False
                else:
                    result = get_result_future.result().result
                    
                    # Check result
                    if result.error_code == FollowCartesianTrajectory.Result.SUCCESSFUL:
                        self.get_logger().debug("move_circular_time_freq: Cartesian trajectory completed successfully")
                        success = True
                    else:
                        self.get_logger().error(f"move_circular_time_freq: Cartesian trajectory failed with error code {result.error_code}: {result.error_message}")
                        success = False

        if block: # Wait for trajectory to complete.
            self.wait_idle(timeout_s=10)
        
        final_elapsed = time.time() - start_time

        if success:
            self.get_logger().debug("move_circular_time_freq: Timed circular motion complete.")          
        else:
            # If motion failed, still try to move up from current position
            self.get_logger().warn("move_circular_time_freq: Circular motion failed, will try to manually move up from current position.")
            self.wait_idle(timeout_s=1) # Wait for robot to be idle in case blocking was not sucessful

            if not self.move_lin_rel_wrf(0.0, 0.0, move_z_distance_mm, 0.0, 0.0, 0.0):
                self.get_logger().error("move_circular_time_freq: Failed to move up after circle failure.")
            else:
                self.get_logger().debug("move_circular_time_freq: Moved up after failure.")

        self.get_logger().info(f"move_circular_time_freq: Timed circular motion finished. Total time in controller: {final_elapsed:.1f}s, Waypoints in trajectory: {len(waypoints)}, Success: {success}")
        return success
    
    def move_pose_and_wait(self, x: float, y: float, z: float, a: float, b: float, g: float, 
                           verify: bool = True, timeout_s: float = 20.0,
                           pos_tol_mm: float = 0.5, orient_tol_deg: float = 0.5, vel_threshold: float = 0.1):
        """ Sends MovePose and waits for the robot to reach the target pose. Verifies the target.
        
        Inputs:
            - x: Target x position in mm.
            - y: Target y position in mm.
            - z: Target z position in mm.
            - a: Target a position in degrees.
            - b: Target b position in degrees.
            - g: Target g position in degrees.
            - verify: Whether to verify the target position.
            - timeout_s: Timeout in seconds.
            - pos_tol_mm: Position tolerance in mm.
            - orient_tol_deg: Orientation tolerance in degrees.
            - vel_threshold: Velocity threshold in mm/s.
        """
        self.get_logger().info(f"move_pose_and_wait: Executing MovePose to ({x:.1f},{y:.1f},{z:.1f}) blocking.")
        
        if not self.move_pose(x, y, z, a, b, g):
            self.get_logger().error("move_pose_and_wait: Failed to send MovePose command.")
            return False

        self.wait_idle(timeout_s=timeout_s)
        
        if verify:
            position_reached = False
            for i in range(3): # Check up to 3 times.
                if self.verify_position(x, y, z, a, b, g, 
                                        pos_tol_mm=pos_tol_mm, orient_tol_deg=orient_tol_deg, 
                                        vel_threshold=vel_threshold):
                    position_reached= True
                    break
            
            if position_reached:
                self.get_logger().debug("move_pose_and_wait: Target reached.")
                return True
            else:
                self.get_logger().debug("move_pose_and_wait: Target not reached.")
                return False
        else:
            self.get_logger().debug("move_pose_and_wait: Robot idle.")
            return True

    @classmethod
    def create_pose(cls, x: float, y: float, z: float, 
                         xq: float, yq: float, zq: float, wq: float = 1.0) -> Pose:
        """ Create a pose message from x, y, z, xq, yq, zq, wq.

        Be aware that sometimes we misuse the Pose message as a Pose using euler angles. (xq,yq,zq = a,b,g)
        """
        pose = Pose()
        pose.position.x = x
        pose.position.y = y
        pose.position.z = z
        pose.orientation.x = xq
        pose.orientation.y = yq
        pose.orientation.z = zq
        pose.orientation.w = wq
        return pose

    def spin_until_timer(self, duration_seconds: float):
        """ Spin until a timer expires, allowing callbacks to run.
        
        Inputs:
            - duration_seconds: Time to spin in seconds
        """
        timer_future = rclpy.task.Future()
        timer = None

        def timer_callback():
            timer.cancel()
            timer_future.set_result(True)
        
        timer = self.create_timer(duration_seconds, timer_callback)
        rclpy.spin_until_future_complete(self, timer_future)
        timer.destroy()

    def spin_for_duration(self, duration_seconds: float):
        """ Spin for a specified duration, allowing callbacks to run. (Includes a loop)

        Prefer to use spin_until_timer instead.
        
        Inputs:
            - duration_seconds: Time to spin in seconds
        """
        start_time = time.time()
        while rclpy.ok() and (time.time() - start_time) < duration_seconds:
            rclpy.spin_once(self, timeout_sec=0.01)  # Small timeout to allow callbacks
            time.sleep(1e-4)