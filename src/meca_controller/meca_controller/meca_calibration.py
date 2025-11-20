#!/usr/bin/env python3Publisher

'''
############################################################
############################################################
Purpose: This node, Meca_Control, was created to integrate/test motion planning
with the real robot by communicating over topics and with services. It
can be used to control both robots--hence it doesn't take a namespace.
Currently, "user" code can be written in the run() method to use this
ros2 system and move the robots. In the future, it would be great to
create a shell command prompt for the user to control the robots or get
info quickly without writing code; also, a way to create a separate script
that interacts with these control functions would be ideal so this file does
not get messy. But for now, this is the state of the codebase.

Date Created: 04/26/2022
Developers: Jessica Myers, [add name here]
University of Illinois Urbana-Champaign

############################################################
############################################################
'''

import rclpy # python library for ros2. needed for every ros2 node
from rclpy.node import Node
import mecademicpy.robot as mdr
from geometry_msgs.msg import Pose, PoseStamped, Quaternion
from sensor_msgs.msg import JointState
from custom_interfaces.srv import GetMotionPlan, VisualizeMotionPlan, MoveJoints
from custom_interfaces.srv import GoToPose, MoveGripper, SetBlending, SetGripperForce, SetGripperVel, SetJointVel, SetJointAcc, MoveLin, MoveLinRelWrf, MovePoseEuler, WaitIdle, GetRtJointTorq
from custom_interfaces.msg import RobotStatus, GripperStatus
import numpy as np
import math
from functools import partial
import time
from meca_controller.meca_settings import ROBOT1 # ADJUST THESE GLOBAL CONSTANTS

from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import MotionPlanRequest, WorkspaceParameters, RobotState, Constraints, JointConstraint, PlanningOptions, PositionIKRequest
from moveit_msgs.srv import GetPositionIK 
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
import tf2_ros
import tf2_geometry_msgs
from scipy.spatial.transform import Rotation as R
import traceback
from builtin_interfaces.msg import Duration as RosDurationMsg

# never use scientific notation, always 3 digits after the decimal point
np.set_printoptions(suppress=True, precision=3) 

class Meca_Control(Node):
    def __init__(self):
        super().__init__("meca_control")

        # Subscribe to keep up to date on current joint angles, pose, and status of the robot (may not need all these):
        #self.joint_subscriber_robot1_ = self.create_subscription(JointState, f"{ROBOT1['namespace']}/MecademicRobot_joint_fb",
                                                                 #self.update_robot1_joints_callback, 10)
        self.joint_subscriber_robot1_ = self.create_subscription(JointState, "/joint_states",
                                                                 self.update_robot1_joints_callback, 10)
        self.pose_subscriber_robot1_ = self.create_subscription(Pose, f"{ROBOT1['namespace']}/pose",
                                                                 self.update_robot1_pose_callback, 10)
        self.gripper_subscriber_robot1_ = self.create_subscription(GripperStatus, f"{ROBOT1['namespace']}/gripper_status",
                                                                 self.update_robot1_gripper_status_callback, 10)
        self.status_subscriber_robot1_ = self.create_subscription(RobotStatus, f"{ROBOT1['namespace']}/robot_status",
                                                                 self.update_robot1_status_callback, 10)

        # Initialize variables that will be set later:
        self.current_joints_robot1 = None
        self.current_pose_robot1 = None
        self.current_gripper_status_robot1 = None
        self.current_robot1_status = None
        
        # Add MoveIt2 interface
        self.callback_group = ReentrantCallbackGroup()
        
        # Action client for MoveIt2
        self.moveit_action_client = ActionClient(
            self,
            MoveGroup,
            'move_action',
            callback_group=self.callback_group
        )
        
        # TF buffer for coordinate transformations
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        # Add IK Service Client
        self.ik_client = self.create_client(GetPositionIK, '/compute_ik')
        
        # Wait for MoveIt to be available
        if not self.moveit_action_client.wait_for_server(timeout_sec=2.0):
            self.get_logger().warn('MoveIt action server not available, proceeding without it.')
            self.moveit_available = False
        else:
            self.get_logger().info('MoveIt action server found and ready.')
            self.moveit_available = True

        self.meca_arm_joint_names = [
            "meca_axis_1_joint", "meca_axis_2_joint", "meca_axis_3_joint",
            "meca_axis_4_joint", "meca_axis_5_joint", "meca_axis_6_joint"
        ]

        self.wait_idle_client = self.create_client(WaitIdle, f"{ROBOT1['namespace']}/wait_idle")

    '''
    Purpose: Parts of this code rely upon getting information from topics published to by the meca_driver, such as current_joints.
             Errors will occur if the main code is executed before these are assigned (e.g. motion plan but not yet received the
             current joint states from the subscriber). This function will wait until the critical variables are set so that all
             the code runs as intended.

             -timeout_length: amount of time in seconds that should wait to receive data from both robots before beginning code execution;
                        after timeout period, just waits for data from one robot to begin because assumes you are only using 1 robot.

             TODO add to the critical init variables as code is built.
    '''
    def wait_for_initialization(self, timeout_length=2):
        time_start = time.time()
        while rclpy.ok():
            print('waiting to receive meca state data...')
            rclpy.spin_once(self) # spin once while we wait to avoid blocking any callbacks. (like the data we are waiting on)

            if (self.current_joints_robot1 is not None):
                print('...received data for robot, starting now.')
                return

            # If timeout period has elapsed, only check for data being received from one robot before starting:
            if ((time.time() - time_start) >= timeout_length) and (self.current_joints_robot1 is not None):
                print('...received data for at least one robot, starting now because timeout has occurred waiting.\n')
                return

    '''
    Purpose: A space to write user code to control the robots. Example commands given here.
    
    TODO in the future, create a better way for the user to add code that doesn't involve editing this file.
    Currently unsure how to do this; moved older test code from this function into tests_meca_control.py,
    as a comment.

    NOTE: Everything involving the pinocchio collision detection / urdf / visualization / motion planning has joint
    angles in RADIANS. The mecademicpy interface uses DEGREES for the MoveJoints command.
    '''
    def run(self):
        #pose14 = [-7.278, 264.617, 232.032, -109.232, -1.666, 89.419]
        pose15 = [174.773, 224.915, 129.883, -137.110, 16.966, -146.331]
        # pose16 = [155.901, 44.067, 289.228, -136.207, 39.933, -133.982]
        # pose17 = [-146.705, 69.686, 321.622, -104.114, -48.440, 3.717]
        pose5 = [-13.469, 43.285, 409.894, -77.294, -10.495, 95.054]  
        #pose6 = [-8.031, 71.561, 382.850, -78.296, -6.270, 91.282]
        pose7 = [-82.579, 103.605, 365.126, -87.108, -29.984, 36.562]
        #pose8 = [-192.097, 142.046, 250.900, -126.319, -32.127, 2.206]
        #pose9 = [-218.350, 79.220, 207.748, -145.781, -34.376, -22.875]
        #pose10 = [44.877, 31.487, 341.816, -84.372, -20.214, 41.473]
        pose11 = [-23.422, 104.944, 334.296, -94.552, -10.792, 79.425]
        pose12 = [-44.691, 33.257, 239.214, -108.302, -56.465, 35.045]
        pose13 = [190.336, 162.313, 228.349, -130.038, 21.687, -147.733]
        #pose1 = [-19.430, 249.801, 285.035, -108.403, -5.271, 91.375]
        pose2 = [-17.694, 249.814, 183.603, -108.707, -3.845, 33.316]
        pose3 = [161.813, 154.199, 69.148, -160.085, 16.162, -175.931]
        pose4 = [-7.278, 264.617, 282.032, -99.232, -1.666, 89.419]
        pose18 = [-92.115, 297.068, 157.690, -122.805, -10.012, 96.789]
        # pose19 = [-137.535, 280.211, 114.698, -128.689, -10.039, 72.987]
        # pose20 = [-212.463, 215.875, 104.623, -137.609, -14.713, 54.770]
        list_of_poses = [pose11, pose12, pose13, pose4, pose15, pose7,
                         pose2, pose3,
                         pose5, pose18] 


        # Safety config
        self.set_joint_vel(ROBOT1['namespace'], 20)
        self.set_gripper_force(ROBOT1['namespace'], 5)
        self.move_joints(np.array([0, 0, 0, 0, 0, 0]), ROBOT1['namespace'])

        for i in range(len(list_of_poses)):
            print(f"Moving to pose {i+1}...")
            pose_reached = self.move_pose_and_wait(ROBOT1['namespace'], list_of_poses[i][0], list_of_poses[i][1], list_of_poses[i][2], list_of_poses[i][3], list_of_poses[i][4], list_of_poses[i][5])
            print(f"Robot at pose {i+1}. Check Charuco detection in rqt_image_view.")
            print("When detection is good, click 'Take Sample' in the RViz moveit_calibration panel.")
            input("Then, press Enter here to move to the next pose...")
        
        self.move_joints(np.array([0, 0, 0, 0, 0, 0]), ROBOT1['namespace'])

    

    def update_robot1_joints_callback(self, joints):
        self.current_joints_robot1 = joints # JointState (has .position and .velocity fields)

    def update_robot1_pose_callback(self, pose_msg):
        self.current_pose_robot1 = pose_msg # Pose (has .position and .orientation fields)
    
    def update_robot1_gripper_status_callback(self, gripper_status):
        self.current_gripper_status_robot1 = gripper_status # custom msg type

    def update_robot1_status_callback(self, robot_status):
        self.current_robot1_status = robot_status # custom msg type


    '''
    Purpose: used to determine from the namespace whether this is robot1 or robot2, for the purpose of getting the order right
             in the len 12 joint angle array sent to the motion planner and visualization.
    '''
    def determine_which_robot_from_namespace(self, namespace):
        is_robot1 = True
        return is_robot1


    '''
    Inputs: required: namespace of the robot for which you absolutely must know the current configuration in order to do motion
                      planning, because you are going to execute motion for that robot. Exception will be thrown if the current
                      joint angles are not being received for this robot.
                            - Ideally, you want to be receiving the joint angles for BOTH robots--a warning will be printed to
                              proceed with caution if the other robot's angles are not being received, but motion planning will
                              continue (at your own risk), using default resting config for the other robot for collision detection.

    Returns a np array of size 6 specifying the joint angles of the robot. If the robot's current joints are none,
    then the robot is not turned on and we are not currently receiving the joint angles. The default angles will be set to
    the robot's tpyical resting pose (feel free to change if necessary). Use motion planning at your own risk if you do not
    know the state of the other robot.

    '''
    def get_current_joints_both_robots(self, required):
        # 1) Figure out if this is robot1 or robot2:
        is_robot1 = self.determine_which_robot_from_namespace(required)

        # 2) Make sure receiving current joint angles from the required robot, print warning if this robot isn't required:
        if self.current_joints_robot1 is None:
            if is_robot1:
                raise Exception(f"ERROR in get_current_joints_both_robots: Not actively receiving current joint state from robot1; check if it is on and publishing.")
            else:
                self.get_logger().warn('Not actively receiving current joint state from robot1; check if it is on and publishing.' \
                            ' Use motion planning at your own risk. Defaulting to ASSUMED_RESTING_STATE_CONFIG for the robot1' \
                            ' during planning.\n\n')            
                # robot1_joints = ASSUMED_RESTING_STATE_CONFIG
        else:
            robot1_joints = self.current_joints_robot1.position

        # 3) Get len 6 joint angles
        return robot1_joints
    
    '''
    Service call

    Inputs:
    - desired_joint_angles (degrees): np.array of len 6.
    - robot_namespace: the namespace of the robot which you want to control. Should be /robot1 or /robot2
    - error_tolerance: (degrees) the joint angle tolerance within which the robot should reach before executing the next command
    - timeout_length: the max amount of time in seconds to wait on the robot to reach the desired position before returning failure

    Returns:
    - is_reached: boolean, whether the robot successfully reached the desired position.
    '''
    def move_joints(self, desired_joint_angles, robot_namespace, error_tolerance=.1, timeout_length=60):
        # 1) Set up client
        client = self.create_client(MoveJoints, f'{robot_namespace}/move_joints')
        while not client.wait_for_service(timeout_sec=1.0): # if still waiting on service for 1 sec print
            self.get_logger().warn("service not available, waiting again...")
        
        # 2) Formulate request
        request = MoveJoints.Request()  
        request.requested_joint_angles = desired_joint_angles.astype(np.float64).tolist()

        # 3) Make async call (don't block thread using synchronous call)
        future = client.call_async(request)
        
        # 4) Wait for the request to be processed, but keep spinning to avoid locking up the thread:
        rclpy.spin_until_future_complete(self, future) # note that we don't really care about the response (future.result()) itself.

        # 5) Wait for position to be reached, within a tolerance, and under the timeout constraints (to prevent infinite while loops):
        is_robot1 = self.determine_which_robot_from_namespace(robot_namespace)
        time_start = time.time()
        while rclpy.ok():
            rclpy.spin_once(self) # spin once while we wait to avoid blocking any callbacks. (example: joint angle updates)
            
            if (time.time() - time_start) >= timeout_length:
                return False # was not reached to the desired precision within the specified amount of time
            
            # Break out when reaches desired joint angles:
            # if self.current_joints is not None:

            if self.has_reached_config(desired_joint_angles,
                                       self.current_joints_robot1.position,
                                       error_tolerance):
                #self.get_logger().info(f"move_joints: Has reached desired joint angles position {desired_joint_angles}")
                return True # has reached the desired position
        
        return False # some undefined behavior occurred; should never reach here
    
    '''
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
    '''
    def move_gripper(self, robot_namespace, position, command="pos"):
        # 1) Set up client
        client = self.create_client(MoveGripper, f'{robot_namespace}/move_gripper')
        while not client.wait_for_service(timeout_sec=1.0): # if still waiting on service for 1 sec print
            self.get_logger().warn("MoveGripper service not available, waiting again...")
        
        # 2) Formulate request
        request = MoveGripper.Request()  
        request.command = command
        request.pos = float(position)

        # 3) Make async call (don't block thread using synchronous call)
        future = client.call_async(request)
        
        # 4) Wait for the request to be processed, but keep spinning to avoid locking up the thread:
        rclpy.spin_until_future_complete(self, future) # note that we don't really care about the response (future.result()) itself.

        # 5) Process response:
        response = future.result()
        if response.error:
            raise Exception(f"ERROR in MoveGripper call: either incorrect command or position out of range. See driver for specific error message.")
        self.get_logger().info(f"MoveGripper for {robot_namespace} complete.")
        return True
    def open_gripper(self, robot_namespace):
        self.move_gripper(robot_namespace, 0, command="open")

    def close_gripper(self, robot_namespace):
        self.move_gripper(robot_namespace, 0, command="close")
    
    def move_lin_rel_wrf(self, robot_namespace, x, y, z, alpha, beta, gamma):
        client = self.create_client(MoveLinRelWrf, f'{robot_namespace}/move_lin_rel_wrf')
        while not client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn("MoveLinRelWrf service not available, waiting again...")
        request = MoveLinRelWrf.Request()
        request.x_offset = x
        request.y_offset = y
        request.z_offset = z
        request.alpha = alpha
        request.beta = beta
        request.gamma = gamma
        future = client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        if future.result() is not None:
            if hasattr(future.result(), 'success'):
                if future.result().success:
                    self.get_logger().info("MoveLinRelWrf completed successfully.")
                    return True
                else:
                    self.get_logger().error("MoveLinRelWrf failed.")
                    return False
            else:
                self.get_logger().info("MoveLinRelWrf command sent (no response flag).")
                return False
        else:
            self.get_logger().error("No response received from MoveLinRelWrf service.")
            return False
    def move_lin(self, robot_namespace, x, y, z, alpha, beta, gamma):
        client = self.create_client(MoveLin, f'{robot_namespace}/move_lin')
        while not client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn("MoveLin service not available, waiting again...")
        request = MoveLin.Request()
        request.x = float(x)
        request.y = float(y)
        request.z = float(z)
        request.alpha = float(alpha)
        request.beta = float(beta)
        request.gamma = float(gamma)
        try:
            self.get_logger().info(f"Calling MoveLin service for ({x:.1f}, {y:.1f}, {z:.1f})...")
            future = client.call_async(request)
            rclpy.spin_until_future_complete(self, future, timeout_sec=70.0) # Add timeout (slightly > driver WaitIdle)
            
            result = future.result()

            if result is not None:
                # Check if the service definition HAS a success field
                if hasattr(result, 'success'): 
                    if result.success:
                        self.get_logger().info("MoveLin service reported SUCCESS.")
                        return True # Explicit success
                    else:
                        err_msg = getattr(result, 'error_message', 'No error message provided.')
                        self.get_logger().error(f"MoveLin service reported FAILURE: {err_msg}")
                        return False # Explicit failure
                else: # No success field in .srv definition
                    self.get_logger().info("MoveLin command sent (service has no success field, assuming success).")
                    return True # Assume success if call completed without exception
            else:
                # This happens if spin_until_future_complete times out or service died
                self.get_logger().error("No response received from MoveLin service (timed out or service died?).")
                return False 

        except Exception as e:
            self.get_logger().error(f"Exception during MoveLin service call: {e}")
            import traceback
            traceback.print_exc()
            return False
        
    def move_pose(self, robot_namespace, x, y, z, alpha, beta, gamma):
        """Sends a MovePose command and waits for completion."""
        # Use the correct service name matching the driver
        client = self.create_client(MovePoseEuler, f'{robot_namespace}/move_pose') 
        if not client.wait_for_service(timeout_sec=0.1): # Quick check if service is up
            self.get_logger().error(f"MovePose service not available quickly for {robot_namespace}.")
            return False # Cannot send
        
        request = MovePoseEuler.Request() # Match service type
        request.x = float(x)
        request.y = float(y)
        request.z = float(z)
        request.alpha = float(alpha)
        request.beta = float(beta)
        request.gamma = float(gamma)

        future = client.call_async(request)
        self.get_logger().debug(f"MovePose command for ({x:.1f}, {y:.1f}, {z:.1f}) SENT to driver.")
        return True # Assume sent if no immediate exception
        
        '''
        try:
            self.get_logger().info(f"Calling MovePose service for ({x:.1f}, {y:.1f}, {z:.1f})...")
            future = client.call_async(request)
            # Add a longer timeout here as MovePose can take time
            rclpy.spin_until_future_complete(self, future, timeout_sec=70.0) 
            
            result = future.result()

            if result is not None:
                if result.success:
                    self.get_logger().info("MovePose service reported SUCCESS.")
                    return True
                else:
                    self.get_logger().error(f"MovePose service reported FAILURE: {result.error_message}")
                    return False
            else:
                self.get_logger().error("No response received from MovePose service (timed out or service died?).")
                return False 

        except Exception as e:
            self.get_logger().error(f"Exception during MovePose service call: {e}")
            import traceback
            traceback.print_exc()
            return False
        '''
        
    def get_rt_joint_torq(self, robot_namespace, include_timestamp, synchronous_update, timeout):
        client = self.create_client(GetRtJointTorq, f'{robot_namespace}/get_rt_joint_torq')
        while not client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn("GetRtJointTorq service not available, waiting...")
        request = GetRtJointTorq.Request()
        request.include_timestamp = include_timestamp
        request.synchronous_update = synchronous_update
        request.timeout = timeout
        future = client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        if future.result() is not None and future.result().success:
            self.get_logger().info(f"Received joint torques: {future.result().torques}")
            return future.result()
        else:
            self.get_logger().error("Failed to get joint torques")
            return None
    
    '''
    Purpose: Used to see whether the robot has reached its desired configuration, to a tolerance (in degrees).
    '''
    def has_reached_config(self, desired_joint_angles, current_joint_angles, error_tolerance=.1):
        # Convert desired angles from degrees to radians
        desired_radians = np.array(desired_joint_angles) * (np.pi / 180.0)
        # Take the AND operation across the list of booleans
        #self.get_logger().info(f"waiting for {desired_joint_angles} degrees ({desired_radians} rad), currently it is {current_joint_angles} rad")
        return np.all([self.within_tolerance(desired_rad, curr_joint, error_tolerance) 
                    for (desired_rad, curr_joint) in zip(desired_radians, current_joint_angles)])
    
    # t - .0001 <= x <= t + .0001:
    def within_tolerance(self, desired_joint_pos, current_joint_pos, error_tolerance):
        return (((desired_joint_pos - error_tolerance) <= current_joint_pos) and 
                (current_joint_pos <= (desired_joint_pos + error_tolerance)))
    
    # Helper: Convert a configuration (numpy array in radians) to a RobotState message
    def create_robot_state_from_config(self, config):
        rs = RobotState()
        js = JointState()
        js.name = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]
        js.position = config.tolist()
        rs.joint_state = js
        return rs

    def create_goal_constraints(self, target_joints_rad):
        """
        Create goal constraints based on target joint angles.
        This function constructs a Constraints message containing JointConstraint entries
        for each joint in the planning group ("meca_arm"). It assumes your robot has six joints.
        
        Args:
            target_joints_rad: NumPy array of target joint positions (in radians).
        
        Returns:
            A list containing one Constraints message.
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

    def plan_to_joint_target(self, joint_angles):
        """
        Plans a trajectory to reach the desired joint angles using MoveIt's planning action.
        Assumes joint_angles is a NumPy array in degrees.
        
        Returns:
            (success: bool, trajectory): The success flag and the planned trajectory.
        """
        if not self.moveit_available:
            self.get_logger().error('MoveIt is not available.')
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
        send_goal_future = self.moveit_action_client.send_goal_async(goal_msg)
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



    '''
    Service call

    Request inputs:
        - blending [float] from 0 to 100
    
    Response:
        - error [bool]: True if error occurred, False otherwise.
    '''
    def set_blending(self, robot_namespace, blending):
        """Sets blending and returns True on success, False on failure."""
        client = self.create_client(SetBlending, f'{robot_namespace}/set_blending')
        if not client.wait_for_service(timeout_sec=1.0):
            self.get_logger().error("SetBlending service not available.")
            return False # Service not found

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
                        self.get_logger().info(f"SetBlending to {blending} for {robot_namespace} complete.")
                        return True # Explicit success
                    else:
                        self.get_logger().error(f"SetBlending service reported an error (value likely out of range: {blending}).")
                        return False # Explicit failure reported by service
                else: # No error field in .srv definition
                    self.get_logger().info(f"SetBlending to {blending} command sent (no error field in response, assuming success).")
                    return True # Assume success if call completed
            else:
                self.get_logger().error("No response received from SetBlending service (timed out?).")
                return False

        except Exception as e:
            self.get_logger().error(f"Exception during SetBlending service call: {e}")
            import traceback
            traceback.print_exc()
            return False


    def wait_for_motion_complete(
        self,
        target_joints_deg: np.ndarray,
        tol_deg: float = 0.5,
        timeout_s: float = 60.0,
        vel_threshold: float = 0.001,
    ) -> bool:
        """
        Blocks until the robot has reached `target_joints_deg` (deg)
        AND joint velocities are below vel_threshold (rad/s),
        or until timeout_s elapses.  Then sends one final MoveJoints
        service call and waits for it to complete.
        """
        start = time.time()
        last_actual = None

        # 1) Wait for pose + velocity condition
        while rclpy.ok() and (time.time() - start) < timeout_s:
            rclpy.spin_once(self, timeout_sec=0.02)

            js = self.current_joints_robot1
            if js is None:
                continue

            actual_deg = np.degrees(np.array(js.position[:6]))
            vel       = np.array(js.velocity[:6])
            last_actual = actual_deg

            # debug
            #self.get_logger().info(f"actual deg {actual_deg} and vel {vel}")

            if (
                np.allclose(actual_deg, target_joints_deg, atol=tol_deg)
                and np.all(np.abs(vel) < vel_threshold)
            ):
                break

        if last_actual is None:
            self.get_logger().warn("Never saw any joint state, aborting wait.")
            return False

        if not np.allclose(last_actual, target_joints_deg, atol=tol_deg):
            self.get_logger().warn(f"Timed out waiting for motion to {target_joints_deg.tolist()}")
            return False

        # 2) FINAL "NUDGE" via MoveJoints service
        self.get_logger().info(f"actual deg {actual_deg} and vel {vel}")
        self.get_logger().info(f"Robot idle near target; sending nudge to {target_joints_deg.tolist()}")

        # --- set up service client ---
        client = self.create_client(MoveJoints, f"{ROBOT1['namespace']}/move_joints")
        if not client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error("MoveJoints service never came up for nudge")
            return False

        req = MoveJoints.Request()
        req.requested_joint_angles = target_joints_deg.tolist()
        fut = client.call_async(req)
        rclpy.spin_until_future_complete(self, fut, timeout_sec=timeout_s - (time.time() - start))
        if fut.result() is None:
            self.get_logger().error("Final nudge MoveJoints call failed or timed out")
            return False

        # 3) Give it a moment to actually report the new state
        end_wait = time.time() + 0.2
        while rclpy.ok() and time.time() < end_wait:
            rclpy.spin_once(self, timeout_sec=0.02)
        
        js_fin = self.current_joints_robot1
        actual_deg_fin = np.degrees(np.array(js_fin.position[:6]))
        vel_fin       = np.array(js_fin.velocity[:6])
        self.get_logger().info(f"Fine-tuned by {target_joints_deg-actual_deg}")
        self.get_logger().info(f"Final: deg {actual_deg_fin} and vel {vel_fin}")
        return True


    '''
    Service call

    Request inputs:
        - gripper_force: from 5 to 100, which is a percentage of the maximum force the MEGP 25E gripper can hold (40N).
    
    Response:
        - error [bool]: True if error occurred, False otherwise.
    '''
    def set_gripper_force(self, robot_namespace, gripper_force):
        # 1) Set up client
        client = self.create_client(SetGripperForce, f'{robot_namespace}/set_gripper_force')
        while not client.wait_for_service(timeout_sec=1.0): # if still waiting on service for 1 sec print
            self.get_logger().warn("SetGripperForce service not available, waiting again...")
        
        # 2) Formulate request
        request = SetGripperForce.Request()  
        request.gripper_force = float(gripper_force)

        # 3) Make async call (don't block thread using synchronous call)
        future = client.call_async(request)
        
        # 4) Wait for the request to be processed, but keep spinning to avoid locking up the thread:
        rclpy.spin_until_future_complete(self, future) # note that we don't really care about the response (future.result()) itself.

        # 5) Process response:
        response = future.result()
        if response.error:
            raise Exception(f"ERROR: SetGripperForce only accepts a float in range [5, 100]. You gave {gripper_force}.")
        self.get_logger().info(f"SetGripperForce to {gripper_force} for {robot_namespace} complete.")


    '''
    Service call

    Request inputs:
        - gripper_vel: from 5 to 100, which is a percentage of the maximum finger velocity of the MEGP 25E gripper (∼100 mm/s).
    
    Response:
        - error [bool]: True if error occurred, False otherwise.
    '''
    def set_gripper_vel(self, robot_namespace, gripper_vel):
        # 1) Set up client
        client = self.create_client(SetGripperVel, f'{robot_namespace}/set_gripper_vel')
        while not client.wait_for_service(timeout_sec=1.0): # if still waiting on service for 1 sec print
            self.get_logger().warn("SetGripperVel service not available, waiting again...")
        
        # 2) Formulate request
        request = SetGripperVel.Request()  
        request.gripper_vel = float(gripper_vel)

        # 3) Make async call (don't block thread using synchronous call)
        future = client.call_async(request)
        
        # 4) Wait for the request to be processed, but keep spinning to avoid locking up the thread:
        rclpy.spin_until_future_complete(self, future) # note that we don't really care about the response (future.result()) itself.

        # 5) Process response:
        response = future.result()
        if response.error:
            raise Exception(f"ERROR: SetGripperVel only accepts a float in range [5, 100]. You gave {gripper_vel}.")
        self.get_logger().info(f"SetGripperVel to {gripper_vel} for {robot_namespace} complete.")
        
    

    '''
    Service Call

    Request inputs:
        - joint_vel: from 0.001 to 100, which is a percentage of maximum joint velocities.
            - NOTE while you can specify the velocity as .001, I do not recommend it, as it was so slow I did not visually see
              movement. 1 works pretty well for moving slowly; I also do not recommend 100, as that is dangerously fast.
            - NOTE see the meca programming manual (https://cdn.mecademic.com/uploads/docs/meca500-r3-programming-manual-8-3.pdf)
              for more information; the velocities of the different joints are set proportionally as a function
              of their max speed.
    Response:
        - error [bool]: True if error occurred, False otherwise.
    '''
    def set_joint_vel(self, robot_namespace, joint_vel):
        # 1) Set up client
        client = self.create_client(SetJointVel, f'{robot_namespace}/set_joint_vel')
        while not client.wait_for_service(timeout_sec=1.0): # if still waiting on service for 1 sec print
            self.get_logger().warn("SetJointVel service not available, waiting again...")
        
        # 2) Formulate request
        request = SetJointVel.Request()  
        request.joint_vel = float(joint_vel)

        # 3) Make async call (don't block thread using synchronous call)
        future = client.call_async(request)
        
        # 4) Wait for the request to be processed, but keep spinning to avoid locking up the thread:
        rclpy.spin_until_future_complete(self, future) # note that we don't really care about the response (future.result()) itself.

        # 5) Process response:
        response = future.result()
        if response.error:
            raise Exception(f"ERROR: SetJointVel only accepts a float in range [.001, 100]. You gave {joint_vel}.")
        self.get_logger().info(f"SetJointVel to {joint_vel} for {robot_namespace} complete.")

    '''
    Service call

    Request inputs:
        - joint_acc: from 0.001 to 150, which is a percentage of maximum acceleration of the joints, ranging from 0.001% to 150%
    Response:
        - error [bool]: True if error occurred, False otherwise.
    '''
    def set_joint_acc(self, robot_namespace, joint_acc):
        # 1) Set up client
        client = self.create_client(SetJointAcc, f'{robot_namespace}/set_joint_acc')
        while not client.wait_for_service(timeout_sec=1.0): # if still waiting on service for 1 sec print
            self.get_logger().warn("SetJointAcc service not available, waiting again...")
        
        # 2) Formulate request
        request = SetJointAcc.Request()  
        request.joint_acc = float(joint_acc)

        # 3) Make async call (don't block thread using synchronous call)
        future = client.call_async(request)
        
        # 4) Wait for the request to be processed, but keep spinning to avoid locking up the thread:
        rclpy.spin_until_future_complete(self, future) # note that we don't really care about the response (future.result()) itself.

        # 5) Process response:
        response = future.result()
        if response.error:
            raise Exception(f"ERROR: SetJointAcc only accepts a float in range [.001, 150]. You gave {joint_acc}.")
        self.get_logger().info(f"SetJointAcc to {joint_acc} for {robot_namespace} complete.")

    def compute_ik_sync(self, target_pose_stamped: PoseStamped, 
                         start_joints_rad: list[float] | None = None, 
                         group_name: str = "meca_arm") -> tuple[bool, list[float] | None]:
        """Calls MoveIt's /compute_ik service synchronously."""
        if not self.moveit_available or not self.ik_client.service_is_ready(): # Check self.moveit_available
             self.get_logger().error('/compute_ik service not available for compute_ik_sync.')
             return False, None
        
        req = GetPositionIK.Request()
        ik_req = PositionIKRequest() 
        ik_req.group_name = group_name
        ik_req.pose_stamped = target_pose_stamped
        
        # Frame ID for the seed state's joint_state message. Often matches the planning frame.
        # If your planning frame in MoveIt is "meca_base_link", use that. If "world", use "world".
        # This tells MoveIt how to interpret the seed if it involves TF. For direct joint values, it's less critical
        # but good practice to set it to a known frame.
        ik_req.robot_state.joint_state.header.frame_id = target_pose_stamped.header.frame_id 
        
        # Seed state
        if start_joints_rad is not None and len(start_joints_rad) == len(self.meca_arm_joint_names):
             ik_req.robot_state.joint_state.name = self.meca_arm_joint_names
             ik_req.robot_state.joint_state.position = start_joints_rad
             ik_req.robot_state.is_diff = False 
             self.get_logger().info(f"Seeding IK with (rad): {np.array(start_joints_rad).round(3)}")
        else:
             ik_req.robot_state.is_diff = True # Let MoveIt use current robot state as seed if available to it
             self.get_logger().info("No valid seed state provided to compute_ik_sync, MoveIt will use current robot state as seed.")
        
        timeout_val_sec = 0.5 # Define the timeout value for a single IK solve
        timeout_duration_msg = RosDurationMsg()
        timeout_duration_msg.sec = int(math.floor(timeout_val_sec)) # Integer seconds
        timeout_duration_msg.nanosec = int((timeout_val_sec - math.floor(timeout_val_sec)) * 1e9) # Integer nanoseconds
        ik_req.timeout = timeout_duration_msg        
        #ik_req.attempts = 5  # Number of attempts
        ik_req.avoid_collisions = True # Important

        req.ik_request = ik_req 

        self.get_logger().info(f"Calling IK for Pose in frame '{target_pose_stamped.header.frame_id}': "
                               f"P(x={target_pose_stamped.pose.position.x:.1f}, y={target_pose_stamped.pose.position.y:.1f}, z={target_pose_stamped.pose.position.z:.1f}) "
                               f"O(x={target_pose_stamped.pose.orientation.x:.2f}, y={target_pose_stamped.pose.orientation.y:.2f}, z={target_pose_stamped.pose.orientation.z:.2f}, w={target_pose_stamped.pose.orientation.w:.2f})")

        try:
            future = self.ik_client.call_async(req)
            # Wait for the service call to complete
            # Timeout should be generous enough for all attempts
            service_call_timeout_sec = timeout_val_sec + 0.5 # IK timeout + buffer for communication
            rclpy.spin_until_future_complete(self, future, timeout_sec=service_call_timeout_sec) 
            
            if not future.done():
                 self.get_logger().error("IK service call to /compute_ik timed out waiting for result.")
                 return False, None
                 
            response = future.result()

            if response is None:
                 self.get_logger().error("IK service call to /compute_ik failed (result is None).")
                 return False, None

            if response.error_code.val == response.error_code.SUCCESS:
                # The solution.joint_state.position might contain more joints than just the arm
                # (e.g., if the RobotState includes passive or mimicked joints).
                # We need to extract the values for our specific arm joints.
                solution_joint_names = response.solution.joint_state.name
                solution_joint_positions = response.solution.joint_state.position
                
                arm_joint_solution_rad = [0.0] * len(self.meca_arm_joint_names)
                all_found = True
                for i, name_to_find in enumerate(self.meca_arm_joint_names):
                    try:
                        idx_in_solution = list(solution_joint_names).index(name_to_find)
                        arm_joint_solution_rad[i] = solution_joint_positions[idx_in_solution]
                    except ValueError:
                        self.get_logger().error(f"Joint '{name_to_find}' not found in IK solution's joint names: {solution_joint_names}")
                        all_found = False
                        break
                
                if not all_found:
                    return False, None

                self.get_logger().info(f"IK Solution Found (deg): {np.degrees(arm_joint_solution_rad).round(1)}")
                return True, arm_joint_solution_rad
            else:
                 self.get_logger().warn(f"IK failed with MoveIt error code: {response.error_code.val} "
                                      f"(NO_IK_SOLUTION={response.error_code.NO_IK_SOLUTION}, "
                                      f"TIMED_OUT={response.error_code.TIMED_OUT}, "
                                      f"PLANNING_FAILED={response.error_code.PLANNING_FAILED}) " # Added PLANNING_FAILED
                                      f"for target pose in frame '{target_pose_stamped.header.frame_id}'.") # Log target frame
                 return False, None
        except Exception as e:
            self.get_logger().error(f"Exception during compute_ik_sync service call: {e}")
            traceback.print_exc()
            return False, None
    
    
    def move_circular_ik(self, radius_mm: float, num_points: int, num_laps: int, 
                         robot_namespace: str, clockwise: bool = True, 
                         planning_frame: str = "meca_base_link"): # Added planning_frame
        """
        Moves the robot TCP in N circles using joint space planning via IK.
        Uses your existing BLOCKING self.move_joints.
        """
        self.get_logger().info(f"Starting IK-based circular move (blocking joints): R={radius_mm}mm, Pts={num_points}, Laps={num_laps}")

        if num_points < 3 or num_laps < 1:
             self.get_logger().error("Invalid points or laps for circle.")
             return False

        # --- Get Start State (Pose for center, Joints for first seed) ---
        self.get_logger().info("Getting initial pose and joint state (waiting up to 3s)...")
        start_pose_msg = None
        start_joints_msg = None 
        start_time_wait = self.get_clock().now()
        while rclpy.ok() and (self.get_clock().now() - start_time_wait).nanoseconds < 3e9: 
             rclpy.spin_once(self, timeout_sec=0.1)
             if self.current_pose_robot1 and self.current_joints_robot1 and \
                hasattr(self.current_joints_robot1, 'position') and \
                self.current_joints_robot1.position and \
                len(self.current_joints_robot1.position) >= len(self.meca_arm_joint_names):
                  start_pose_msg = self.current_pose_robot1
                  start_joints_msg = self.current_joints_robot1 
                  self.get_logger().info("Initial pose and joints received.")
                  break
             time.sleep(0.1)

        if start_pose_msg is None or start_joints_msg is None:
             self.get_logger().error("Failed to get valid initial pose AND/OR joint state for circle.")
             return False
             
        # Extract initial values
        cx = start_pose_msg.position.x
        cy = start_pose_msg.position.y
        cz = start_pose_msg.position.z
        ca_deg, cb_deg, cg_deg = start_pose_msg.orientation.x, start_pose_msg.orientation.y, start_pose_msg.orientation.z
        
        # Convert current Euler (degrees) to Quaternion using Scipy
        # IMPORTANT: Verify 'zyx' (Yaw, Pitch, Roll) is correct for your Mecademic Alpha, Beta, Gamma.
        try:
            rot = R.from_euler('zyx', [math.radians(ca_deg), math.radians(cb_deg), math.radians(cg_deg)], degrees=False)
            start_orientation_q_xyzw = rot.as_quat() # Scipy returns [x, y, z, w]
        except Exception as e:
            self.get_logger().error(f"Scipy Euler to Quaternion conversion failed: {e}")
            traceback.print_exc()
            return False

        
        # Initial joints (radians) from current_joints_robot1.position for the arm
        last_ik_solution_rad = list(start_joints_msg.position[:len(self.meca_arm_joint_names)]) 
        initial_joints_deg_for_return = np.degrees(last_ik_solution_rad) 
        '''
        # Inside move_circular_ik, after getting start_pose_msg and start_orientation_q
        self.get_logger().info("--- Testing IK for STARTING POSE ---")
        test_ik_ps = PoseStamped()
        test_ik_ps.header.frame_id = planning_frame # e.g., "meca_base_link"
        test_ik_ps.header.stamp = self.get_clock().now().to_msg()
        test_ik_ps.pose.position.x = cx
        test_ik_ps.pose.position.y = cy
        test_ik_ps.pose.position.z = cz
        test_ik_ps.pose.orientation = Quaternion(x=start_orientation_q_xyzw[0],y=start_orientation_q_xyzw[1],z=start_orientation_q_xyzw[2],w=start_orientation_q_xyzw[3])

        ik_ok, ik_test_joints = self.compute_ik_sync(test_ik_ps, last_ik_solution_rad)
        if not ik_ok:
            self.get_logger().error("IK FAILED for the robot's current reported pose! Check frames/orientation.")
            return False
        else:
            self.get_logger().info(f"IK SUCCESS for current pose. Joints: {np.degrees(ik_test_joints).round(1)}")
        # --- End Test ---
        '''
        
        self.get_logger().info(f"Circle center/orientation (Euler deg): P=({cx:.1f},{cy:.1f},{cz:.1f}), O=({ca_deg:.1f},{cb_deg:.1f},{cg_deg:.1f})")
        self.get_logger().info(f"Initial Seed Joints (deg): {initial_joints_deg_for_return.round(1)}")

        # --- Generate Cartesian Waypoints ---
        cartesian_target_poses_stamped = [] 
        angle_step = 2 * math.pi / num_points
        
        for i in range(1, num_points + 1): 
            theta = i * angle_step
            if clockwise: theta = -theta 
            
            wp_ps = PoseStamped()
            wp_ps.header.frame_id = planning_frame # Use the specified planning frame
            wp_ps.header.stamp = self.get_clock().now().to_msg()
            wp_ps.pose.position.x = cx + radius_mm * math.cos(theta)
            wp_ps.pose.position.y = cy + radius_mm * math.sin(theta)
            wp_ps.pose.position.z = cz 
            wp_ps.pose.orientation = Quaternion(
                x=start_orientation_q_xyzw[0], y=start_orientation_q_xyzw[1], 
                z=start_orientation_q_xyzw[2], w=start_orientation_q_xyzw[3]
            )
            cartesian_target_poses_stamped.append(wp_ps)

        # --- Convert Cartesian Waypoints to Joint Waypoints using IK ---
        joint_waypoints_to_execute_deg = []
        self.get_logger().info("Calculating IK for circle waypoints...")
        for i, target_ps in enumerate(cartesian_target_poses_stamped):
            ik_ok, ik_joint_rad = self.compute_ik_sync(target_ps, last_ik_solution_rad)
            
            if ik_ok and ik_joint_rad:
                joint_waypoints_to_execute_deg.append(np.degrees(ik_joint_rad))
                last_ik_solution_rad = ik_joint_rad # Update seed
            else:
                self.get_logger().error(f"IK failed for circle waypoint {i+1} (Cartesian target: "
                                      f"X:{target_ps.pose.position.x:.1f} Y:{target_ps.pose.position.y:.1f} Z:{target_ps.pose.position.z:.1f}). "
                                      f"Aborting circle.")
                return False # Stop if any IK fails
        
        self.get_logger().info(f"Computed {len(joint_waypoints_to_execute_deg)} joint waypoints via IK.")

        # --- Execute Joint Moves (Using your existing blocking move_joints) ---
        # Blending OFF because move_joints is blocking.
        self.get_logger().info("Setting blending OFF (using blocking MoveJoints)")
        if not self.set_blending(robot_namespace, 0): return False

        overall_motion_success = True
        for lap in range(num_laps):
            self.get_logger().info(f"Starting Lap {lap + 1}/{num_laps}")
            for i, jwp_deg in enumerate(joint_waypoints_to_execute_deg):
                self.get_logger().info(f"  Moving to joint waypoint {i+1}: {np.array(jwp_deg).round(1)}")
                # This is your BLOCKING move_joints call
                if not self.move_joints(np.array(jwp_deg), robot_namespace): 
                     self.get_logger().error(f"MoveJoints failed at lap {lap+1}, waypoint {i+1}.")
                     overall_motion_success = False; break
            if not overall_motion_success: break

        # --- Return to Center (using initial joint state) ---
        if overall_motion_success:
            self.get_logger().info("Circle(s) complete. Returning to center joint position.")
            if not self.move_joints(initial_joints_deg_for_return, robot_namespace): # Blocking
                self.get_logger().error("Failed to return to center joint position.")
                overall_motion_success = False
            else: self.get_logger().info("Returned to center successfully.")
        else: # If circle failed partway, still try to return home
            self.get_logger().warn("Circle motion failed or aborted. Attempting to return to center...")
            if not self.move_joints(initial_joints_deg_for_return, robot_namespace): # Blocking
                self.get_logger().error("Failed to return to center after circle failure.")
            else: self.get_logger().info("Returned to center after failure.")
        
        self.get_logger().info(f"IK-based circular move finished. Overall success: {overall_motion_success}")
        return overall_motion_success

    def move_circular_abs(self, radius_mm: float, num_points: int, num_laps: int, robot_namespace: str, clockwise: bool = True, blending_percent: float = 90.0):
        """
        Moves the robot TCP in N circles in the XY plane (relative to WRF), 
        centered around the robot's current XY position at its current Z height, 
        maintaining the current orientation. Uses blocking absolute MovePose commands.
        Returns to the center (start) position at the end.

        Args:
            radius_mm (float): The radius of the circle in mm.
            num_points (int): How many segments to approximate each circle lap with.
            num_laps (int): How many times to trace the circle.
            robot_namespace (str): Namespace of the target robot (e.g., '/robot1').
            clockwise (bool): Direction of the circle. Default True.
        """
        self.get_logger().info(f"Starting absolute circular move for {robot_namespace}: Radius={radius_mm} mm, Points={num_points}, Laps={num_laps}")

        if num_points < 3:
            self.get_logger().error("Number of points for circle must be at least 3.")
            return False
        if num_laps < 1:
            self.get_logger().warn("Number of laps is less than 1, no circle will be traced.")
            return True 

        # --- Get the starting/center pose ---
        self.get_logger().info("Waiting briefly for current pose feedback...")
        start_pose = None
        for _ in range(10): 
            rclpy.spin_once(self, timeout_sec=0.05)
            if self.current_pose_robot1 is not None:
                start_pose = self.current_pose_robot1
                break 
            time.sleep(0.05)
             
        if start_pose is None:
            self.get_logger().error("Cannot execute circular move: No current pose feedback received.")
            return False
            
        # Store the initial pose values (center and orientation)
        cx = start_pose.position.x
        cy = start_pose.position.y
        cz = start_pose.position.z
        ca = start_pose.orientation.x 
        cb = start_pose.orientation.y
        cg = start_pose.orientation.z 
        
        self.get_logger().info(f"Circle center based on current pose: "
                             f"X={cx:.1f}, Y={cy:.1f}, Z={cz:.1f}, "
                             f"A={ca:.1f}, B={cb:.1f}, G={cg:.1f}")
        
        # --- Generate Waypoints for ONE lap ---
        waypoints = []
        angle_step = 2 * math.pi / num_points
        
        for i in range(1, num_points + 1): 
            theta = i * angle_step
            if clockwise:
                theta = -theta 

            target_x = cx + radius_mm * math.cos(theta)
            target_y = cy + radius_mm * math.sin(theta)
            target_z = cz 
            target_alpha = ca
            target_beta = cb
            target_gamma = cg
            
            waypoints.append((target_x, target_y, target_z, target_alpha, target_beta, target_gamma))
        # The last waypoint brings it back near the start (theta = 2*pi)

        # --- Setup for Blended Motion ---
        self.get_logger().info(f"Setting blending ON to {blending_percent}%")
        if not self.set_blending(robot_namespace, blending_percent): # Use the argument
            self.get_logger().error("Failed to set blending ON. Aborting circular move.")
            return False

        overall_success = True
        # Loop N times
        for lap in range(num_laps):
            self.get_logger().info(f"Queueing Lap {lap + 1}/{num_laps}")
            lap_queued_ok = True
            for i, wp in enumerate(waypoints):
                x, y, z, a, b, g = wp
                
                # Log less frequently for many points/laps
                if (i + lap * num_points) % (num_points // 4 if num_points > 4 else 1) == 0: # Log ~4 times per lap
                    self.get_logger().info(f"  Queueing waypoint for lap {lap+1}, pt {i+1}: ({x:.1f}, {y:.1f}, {z:.1f})")
                # Call the absolute MovePose service client function
                move_queued_ok = self.move_pose(robot_namespace, x, y, z, a, b, g) 
                
                if not move_queued_ok:
                    self.get_logger().error(f"MovePose command failed to queue at lap {lap+1}, waypoint {i+1}. Aborting.")
                    lap_queued_ok = False
                    break 
                # Wait 20ms between sending waypoints
                # Adjust this value. If it's too high, motion won't be smooth.
                # If too low, buffer might still overflow.
                # 0.02s = 50 commands per second. Max Meca command rate is often higher,
                # but ROS service overhead adds up.
                # TODO: make this synchronization automatic
                time.sleep(0.04)

                # No need for extra sleep/spin here because move_pose blocks

            if not lap_queued_ok:
                overall_success = False
                break
        
        # --- Wait for Entire Blended Sequence to Complete ---
        if overall_success and waypoints: # Only wait if commands were likely sent
            self.get_logger().info("All circle waypoints queued. Waiting for physical completion...")
        # Calculate a generous timeout for the robot to finish all queued moves
        # This is highly dependent on speed, number of points, radius etc.
        # Example: (num points * num laps * time_per_segment) + buffer
        # A simpler approach: if each segment is short, maybe N_total_segments * 0.5s + buffer
        estimated_total_motion_time = num_points * num_laps * 0.3 + 5.0 # Rough estimate + buffer
        self.get_logger().info(f"Estimated motion time: {estimated_total_motion_time:.1f}s. Setting WaitIdle timeout.")
        
        wait_successful = self.wait_idle(robot_namespace, timeout_s=estimated_total_motion_time)
        if not wait_successful:
            self.get_logger().error("Robot did not become idle after circular motion sequence.")
            overall_success = False # Mark failure if it doesn't finish

        # --- Return to Center (Point 1) ---
        self.get_logger().info("Setting blending OFF for return to center.")
        if not self.set_blending(robot_namespace, 0):
            self.get_logger().warn("Failed to set blending OFF before returning to center.")
        
        if overall_success: # Only return if the circle(s) seemed okay
            self.get_logger().info("Circle(s) complete. Returning to center position.")
            # Blending is already off
            center_reached = self.move_pose_and_wait(robot_namespace, cx, cy, cz, ca, cb, cg)
            if center_reached:
                self.get_logger().info("Returned to center successfully.")
            else:
                self.get_logger().error("Failed to return to center position.")
                overall_success = False # Mark failure if return move fails
        else:
             # If circle failed, return to center
             self.get_logger().warn("Circle motion failed, attempting to return to center anyway...")
             center_reached = self.move_pose(robot_namespace, cx, cy, cz, ca, cb, cg)
             if center_reached:
                 self.get_logger().info("Returned to center after failure.")
             else:
                 self.get_logger().error("Failed to return to center after circle failure.")
                 overall_success = False

        if overall_success:
            self.get_logger().info("Absolute circular move sequence completed.")
        else:
             self.get_logger().error("Absolute circular move sequence failed.")
             
        return overall_success

    '''
    def move_circular_abs(self, radius_mm: float, num_points: int, num_laps: int, robot_namespace: str, clockwise: bool = True):
        """
        Moves the robot TCP in N circles in the XY plane (relative to WRF), 
        centered around the robot's current XY position at its current Z height, 
        maintaining the current orientation. Uses blocking absolute MovePose commands.
        Returns to the center (start) position at the end.

        Args:
            radius_mm (float): The radius of the circle in mm.
            num_points (int): How many segments to approximate each circle lap with.
            num_laps (int): How many times to trace the circle.
            robot_namespace (str): Namespace of the target robot (e.g., '/robot1').
            clockwise (bool): Direction of the circle. Default True.
        """
        self.get_logger().info(f"Starting absolute circular move for {robot_namespace}: Radius={radius_mm} mm, Points={num_points}, Laps={num_laps}")

        if num_points < 3:
            self.get_logger().error("Number of points for circle must be at least 3.")
            return False
        if num_laps < 1:
            self.get_logger().warn("Number of laps is less than 1, no circle will be traced.")
            return True 

        # --- Get the starting/center pose ---
        self.get_logger().info("Waiting briefly for current pose feedback...")
        start_pose = None
        for _ in range(10): 
            rclpy.spin_once(self, timeout_sec=0.05)
            if self.current_pose_robot1 is not None:
                start_pose = self.current_pose_robot1
                break 
            time.sleep(0.05)
             
        if start_pose is None:
            self.get_logger().error("Cannot execute circular move: No current pose feedback received.")
            return False
            
        # Store the initial pose values (center and orientation)
        cx = start_pose.position.x
        cy = start_pose.position.y
        cz = start_pose.position.z
        ca = start_pose.orientation.x 
        cb = start_pose.orientation.y
        cg = start_pose.orientation.z 
        
        self.get_logger().info(f"Circle center based on current pose: "
                             f"X={cx:.1f}, Y={cy:.1f}, Z={cz:.1f}, "
                             f"A={ca:.1f}, B={cb:.1f}, G={cg:.1f}")
        
        # --- Generate Waypoints for ONE lap ---
        waypoints = []
        angle_step = 2 * math.pi / num_points
        
        for i in range(1, num_points + 1): 
            theta = i * angle_step
            if clockwise:
                theta = -theta 

            target_x = cx + radius_mm * math.cos(theta)
            target_y = cy + radius_mm * math.sin(theta)
            target_z = cz 
            target_alpha = ca
            target_beta = cb
            target_gamma = cg
            
            waypoints.append((target_x, target_y, target_z, target_alpha, target_beta, target_gamma))
        # The last waypoint brings it back near the start (theta = 2*pi)

        # --- Execute Moves (Lap by Lap, Waypoint by Waypoint) ---
        # Blending probably won't have much effect here since each move waits for idle.
        # Set blending off just to be sure.
        self.get_logger().info("Setting blending OFF (WaitIdle used in driver)")
        if not self.set_blending(robot_namespace, 0):
            self.get_logger().error("Failed to set blending OFF. Aborting.")
            return False

        overall_success = True
        # Loop N times
        for lap in range(num_laps):
            self.get_logger().info(f"Starting Lap {lap + 1}/{num_laps}")
            lap_success = True
            for i, wp in enumerate(waypoints):
                x, y, z, a, b, g = wp
                
                self.get_logger().info(f"  Moving to waypoint {i+1}/{len(waypoints)}: ({x:.1f}, {y:.1f}, {z:.1f})")

                # Call the absolute MovePose service client function
                # This call blocks until the robot is idle (due to WaitIdle in driver)
                move_success = self.move_pose(robot_namespace, x, y, z, a, b, g) 
                
                if not move_success:
                    self.get_logger().error(f"MovePose failed at lap {lap+1}, waypoint {i+1}. Aborting circle.")
                    lap_success = False
                    break # Stop this lap if command fails
                
                # No need for extra sleep/spin here because move_pose blocks

            if not lap_success:
                overall_success = False
                break # Stop all laps if one fails

        # --- Return to Center (Point 1) ---
        if overall_success: # Only return if the circle(s) seemed okay
            self.get_logger().info("Circle(s) complete. Returning to center position.")
            # Blending is already off
            center_reached = self.move_pose(robot_namespace, cx, cy, cz, ca, cb, cg)
            if center_reached:
                self.get_logger().info("Returned to center successfully.")
            else:
                self.get_logger().error("Failed to return to center position.")
                overall_success = False # Mark failure if return move fails
        else:
             # If circle failed, return to center
             self.get_logger().warn("Circle motion failed, attempting to return to center anyway...")
             center_reached = self.move_pose(robot_namespace, cx, cy, cz, ca, cb, cg)
             if center_reached:
                 self.get_logger().info("Returned to center after failure.")
             else:
                 self.get_logger().error("Failed to return to center after circle failure.")

        if overall_success:
            self.get_logger().info("Absolute circular move sequence completed.")
        else:
             self.get_logger().error("Absolute circular move sequence failed.")
             
        return overall_success
    '''
    @staticmethod
    def angle_diff(a, b):
        """Calculate the smallest difference between two angles in degrees."""
        # Normalize angles to [0, 360) before diff if necessary, or handle raw diff
        diff = (a - b + 180) % 360 - 180 # Difference in [-180, 180]
        return abs(diff) # We care about the absolute difference for tolerance

    
    def move_pose_and_wait(self, robot_namespace, x, y, z, a, b, g, 
                          timeout_s=20.0, pos_tol_mm=0.5, orient_tol_deg=0.5,
                          vel_threshold=0.1, min_wait_time=0.01):
        """Sends MovePose and waits for the robot to reach the target pose."""
        self.get_logger().info(f"Executing MovePose to ({x:.1f},{y:.1f},{z:.1f}) and waiting...")
        
        # Try to send pose command
        if not self.move_pose(robot_namespace, x, y, z, a, b, g):
            self.get_logger().error("Failed to send MovePose command.")
            return False

        start_time = time.time()
        target_pos = np.array([x, y, z])
        target_orient_deg = np.array([a, b, g])
        last_check_time = 0

        while rclpy.ok() and (time.time() - start_time) < timeout_s:
            rclpy.spin_once(self, timeout_sec=0.02) # Spin frequently with short block
            current_p = self.current_pose_robot1
            
            if current_p:
                curr_pos = np.array([current_p.position.x, current_p.position.y, current_p.position.z])
                curr_orient_deg = np.array([current_p.orientation.x, current_p.orientation.y, current_p.orientation.z])
                
                # Position error
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
                is_still = False
                if self.current_joints_robot1 and self.current_joints_robot1.velocity:
                    joint_vels = np.abs(self.current_joints_robot1.velocity[:6])
                    is_still = np.all(joint_vels < vel_threshold)
                #self.get_logger().info(f"pos_error: {pos_error}, orient_error: {orient_error}, is_still: {is_still}")
                if pos_error <= pos_tol_mm and orient_error <= orient_tol_deg and is_still:
                    self.get_logger().info("MovePoseAndWait: Target reached.")
                    return True

        self.get_logger().error(f"MovePoseAndWait: Timed out waiting to reach target pose.")
        return False
    
    def wait_idle(self, robot_namespace: str, timeout_s: float = 60.0) -> bool:
        """Calls the WaitIdle service and blocks until the robot is idle or timeout."""
        # Use self.wait_idle_client directly
        if not self.wait_idle_client.wait_for_service(timeout_sec=1.0): # Quick check
            self.get_logger().error(f"WaitIdle service for '{robot_namespace}' not available now.")
            return False
        
        request = WaitIdle.Request()
        request.timeout_sec = float(timeout_s) # This is the timeout for the robot's internal WaitIdle
        
        self.get_logger().info(f"Calling WaitIdle service (controller will wait up to {timeout_s + 5.0}s for service response)...")
        future = self.wait_idle_client.call_async(request)
        
        # Spin until the service call itself completes (or this controller-side timeout is hit)
        # The service call will block on the driver side until the robot is idle or driver's timeout.
        rclpy.spin_until_future_complete(self, future, timeout_sec = timeout_s + 5.0) # Controller timeout
        
        result = future.result()
        if result is not None:
            if result.success:
                self.get_logger().info("WaitIdle service reported: Robot is idle.")
                return True
            else:
                self.get_logger().error(f"WaitIdle service reported failure: {result.error_message}")
                return False
        else:
            self.get_logger().error("No response from WaitIdle service (controller timed out waiting for service response).")
            return False



    def grasp_chip_fixed_location_by_lifting(
        self, robot_namespace: str, 
        chip_approach_pose_params: tuple, # (x,y,z_above_surface,a,b,g in degrees)
        initial_press_depth_mm: float = 5.5, # Press down this much past approach to ensure contact
        lift_step_mm: float = 0.1,          # Small steps for lifting to find release
        max_lift_search_mm: float = 3.0,    
        # --- Torque parameters based on your plot (VALUES ARE PERCENTAGES as per plot) ---
        # We'll use Joint 5 as the primary indicator.
        # When pressed, J5 torque is high (e.g., > 25%). When free, it's low (e.g., < 10% and often negative).
        contact_joint_indices: list[int] = [4], # Focus on Joint 5 (index 4)
        torque_delta_release_threshold_percent: float = 0.8, # e.g., if torque drops by >0.8% in one step
        press_torque_increase_threshold_percent: float = 10.0, # Example: J5 must increase by at least 10% from free-air
        # ---------------------------------------------------------------------------------
        grasp_height_above_surface_mm: float = 0.2, # For a 0.5mm thick chip
        gripper_open_pos_mm: float = 18.0,
        gripper_closed_pos_mm: float = 9.5,
        lift_chip_height_mm: float = 5.0
    ) -> bool:
        self.get_logger().info("--- Starting Chip Grasp (Press Down, Lift to Detect Surface) ---")
        approach_x, approach_y, approach_z_start, approach_a, approach_b, approach_g = chip_approach_pose_params

        # 1. Preparations
        if not self.move_gripper(robot_namespace, position=gripper_open_pos_mm): return False
        self.set_blending(robot_namespace, 0)

        # 2. Move to Approach Pose (above expected surface)
        self.get_logger().info(f"Moving to approach pose Z: {approach_z_start:.2f}")
        if not self.move_pose_and_wait(robot_namespace, approach_x, approach_y, approach_z_start, 
                                     approach_a, approach_b, approach_g, timeout_s=15.0):
            self.get_logger().error("Failed to reach approach pose."); return False
        time.sleep(0.3); rclpy.spin_once(self) # Settle
        rt_data_baseline = self.get_rt_joint_torq(robot_namespace, False, True, 1.0)    
        if not rt_data_baseline or not rt_data_baseline.success or len(rt_data_baseline.torques) == 0:
            self.get_logger().error("Failed to get baseline free-air torques."); return False
        baseline_torques_free_air = np.array(rt_data_baseline.torques)
        self.get_logger().info(f"Baseline free-air torques (%): {baseline_torques_free_air.round(2)}")

        # 3. Press Down to Ensure Contact
        self.set_joint_vel(robot_namespace, 2) # Slow and careful for probing
        self.get_logger().info(f"Pressing down by {initial_press_depth_mm:.2f}mm to ensure contact.")
        if not self.move_lin_rel_wrf(robot_namespace, 0.0, 0.0, -initial_press_depth_mm, 0.0, 0.0, 0.0): # Ensure this blocks
             self.get_logger().error("Failed to execute press_down move."); return False
        time.sleep(0.5); rclpy.spin_once(self) 

        rt_data_pressing = self.get_rt_joint_torq(robot_namespace, False, True, 1.0)
        if not rt_data_pressing or not rt_data_pressing.success or len(rt_data_pressing.torques) == 0:
            self.get_logger().error("Failed to get torques while pressing."); return False
        torques_at_full_press = np.array(rt_data_pressing.torques)
        self.get_logger().info(f"Torques (%) while pressing: {torques_at_full_press.round(2)}")

        # Verify significant press on key joint(s)
        pressed_enough = False
        for joint_idx in contact_joint_indices:
            if joint_idx < len(torques_at_full_press):
                if abs(torques_at_full_press[joint_idx]-baseline_torques_free_air[joint_idx])>= press_torque_increase_threshold_percent:
                    pressed_enough = True; break
        if not pressed_enough:
            self.get_logger().error(f"Not enough torque change on contact joints after pressing compared to free-air. Expected values like {press_torque_increase_threshold_percent}%. Aborting.")
            self.move_lin_rel_wrf(robot_namespace, 0.0, 0.0, initial_press_depth_mm, 0.0, 0.0, 0.0) # Move back up
            return False

        # 4. Lift Step-by-Step to Detect Surface Release
        self.get_logger().info(f"Lifting by {lift_step_mm}mm steps to detect surface release via torque change (threshold > {torque_delta_release_threshold_percent}%)...")
        surface_z_at_release = None
        previous_step_torques = np.copy(torques_at_full_press) # Torques when fully pressed
        pose_at_surface_contact = self.current_pose_robot1 # Pose when fully pressed (before starting to lift)
        if not pose_at_surface_contact: # Should have pose here
            self.get_logger().error("Critical: Lost pose feedback before starting lift detection."); return False

        max_lift_steps = int(max_lift_search_mm / lift_step_mm) + 1
        for step_num in range(max_lift_steps):
            # Store pose *before* this specific lift step - this will be the surface Z if release is detected *after* this lift
            pose_before_this_lift_segment = self.current_pose_robot1
            if not pose_before_this_lift_segment:
                self.get_logger().warn(f"No pose feedback before lift step {step_num + 1}. Using last known Z.")
                # Fallback to using Z from pose_at_surface_contact + accumulated lift
                # This can introduce error if a previous pose was missed.
                temp_z = pose_at_surface_contact.position.z + (step_num * lift_step_mm)
                pose_before_this_lift_segment = Pose() # Create a dummy pose
                pose_before_this_lift_segment.position.z = temp_z


            if not self.move_lin_rel_wrf(robot_namespace, 0.0, 0.0, +lift_step_mm, 0.0, 0.0, 0.0): # Blocking
                self.get_logger().error("MoveLinRelWrf lift step failed. Aborting."); return False
            
            time.sleep(0.15) # Settle and allow feedback update
            rclpy.spin_once(self, timeout_sec=0.05)

            rt_data_current = self.get_rt_joint_torq(robot_namespace, False, True, 0.5)
            if not rt_data_current or not rt_data_current.success or len(rt_data_current.torques) == 0:
                self.get_logger().warn("Failed to get current torques during lift search."); continue
            current_torques_percent = np.array(rt_data_current.torques)
            
            released_this_step = False
            for joint_idx in contact_joint_indices:
                 if joint_idx < len(current_torques_percent) and joint_idx < len(previous_step_torques):
                    if joint_idx == 4: # Specifically for Joint 5
                        torque_change_this_step = abs(previous_step_torques[joint_idx] - current_torques_percent[joint_idx])
                        self.get_logger().info(f"Lift step {step_num+1}, J5 Torque: Prev={previous_step_torques[joint_idx]:.2f}%, Curr={current_torques_percent[joint_idx]:.2f}%, Change={torque_change_this_step:.2f}%")
                        if torque_change_this_step < torque_delta_release_threshold_percent:
                            self.get_logger().info(f"SURFACE RELEASE DETECTED on J5! Torque drop {torque_change_this_step:.2f}% < {torque_delta_release_threshold_percent}%")
                            surface_z_at_release = pose_before_this_lift_segment.position.z
                            released_this_step = True; break
            
            if released_this_step: break

            previous_step_torques = np.copy(current_torques_percent) # Update for next iteration
            if (step_num + 1) * lift_step_mm >= max_lift_search_mm: # Check against total distance lifted
                self.get_logger().warn(f"Max lift search ({max_lift_search_mm}mm) reached without clear release via torque delta.")
                # Fallback: assume current pose - step is the surface
                if self.current_pose_robot1:
                     surface_z_at_release = self.current_pose_robot1.position.z - lift_step_mm
                elif pose_before_this_lift_segment: # If current_pose is None but previous was not
                     surface_z_at_release = pose_before_this_lift_segment.position.z
                self.get_logger().warn(f"Assuming surface at Z~{surface_z_at_release:.3f} after max lift.")
                break
        
        # 5. Position for Grasp
        if surface_z_at_release is None:
            self.get_logger().error("Failed to detect box surface by lifting. Aborting grasp.")
            return False

        target_grasp_z = surface_z_at_release + grasp_height_above_surface_mm
        self.get_logger().info(f"Surface detected at Z~{surface_z_at_release:.2f}. Moving to grasp Z={target_grasp_z:.2f}")
        
        if not self.move_pose_and_wait(robot_namespace, approach_x, approach_y, target_grasp_z, 
                                     approach_a, approach_b, approach_g, 
                                     pos_tol_mm=0.1, orient_tol_deg=0.5, timeout_s=10.0): # Tighter tolerance
            self.get_logger().error("Failed to move to final grasp height."); return False
        time.sleep(0.2) # Settle

        # 6. Grasp Chip 
        self.get_logger().info(f"Closing gripper to {gripper_closed_pos_mm}mm")
        if not self.move_gripper(robot_namespace, position=gripper_closed_pos_mm): # Ensure this blocks or add appropriate wait
             self.get_logger().error("Failed to close gripper.")
             return False
        time.sleep(1.0) # Allow gripper to physically close and stabilize grip

        # 7. Lift Chip
        self.get_logger().info(f"Lifting chip by {lift_chip_height_mm}mm")
        if not self.move_lin_rel_wrf(robot_namespace, 0.0, 0.0, lift_chip_height_mm, 0.0, 0.0, 0.0): # Ensure this blocks or waits
             self.get_logger().error("Failed to lift chip.")
             return False
        time.sleep(0.5) # Settle

        # Restore original settings if any were changed (e.g., velocity)
        # if original_vel is not None: self.set_joint_vel(robot_namespace, original_vel)

        self.get_logger().info("--- Chip Grasp by Lifting Sequence Presumed Successful ---")
        return True





    def touchdown_torque_plotter(self, robot_namespace: str,
                                 approach_pose_params: tuple = (190.0, 158.943, 96.577, 0.0, 90.0, 0.0),
                                  step_mm: float = 0.2, 
                                  timeout_s: float = 60.0,
                                  N: int = 10):
        """Plots torque values for specified joints vs Z."""
        import matplotlib.pyplot as plt
        torques_all = []
        dz_values = []
        self.get_logger().info(f"Moving to starting pose...")
        self.move_pose_and_wait(robot_namespace, approach_pose_params[0], approach_pose_params[1], approach_pose_params[2], approach_pose_params[3], approach_pose_params[4], approach_pose_params[5])
        time.sleep(0.5)
        rt_data = self.get_rt_joint_torq(robot_namespace, False, True, 0.5)
        torques_all.append(np.array(rt_data.torques))
        dz_values.append(0.0)  # Initial position
        
        # Move down
        self.get_logger().info(f"Moving down...")
        for i in range(1, N):
            dz = -i * step_mm
            self.move_lin_rel_wrf(robot_namespace, 0.0, 0.0, -step_mm, 0.0, 0.0, 0.0)
            time.sleep(0.1)
            rclpy.spin_once(self, timeout_sec=0.1)
            rt_data = self.get_rt_joint_torq(robot_namespace, False, True, 0.5)
            print(np.array(rt_data.torques))
            torques_all.append(np.array(rt_data.torques))
            dz_values.append(dz)
            
        # Move back up
        self.get_logger().info(f"Moving back up...")
        self.set_joint_vel(robot_namespace, 2) # Slow and careful for probing
        for i in range(1, N):
            dz = i * step_mm
            self.move_lin_rel_wrf(robot_namespace, 0.0, 0.0, step_mm, 0.0, 0.0, 0.0)
            time.sleep(1.0)
            rclpy.spin_once(self, timeout_sec=0.1)
            rt_data = self.get_rt_joint_torq(robot_namespace, False, True, 0.5)
            print(np.array(rt_data.torques))
            torques_all.append(np.array(rt_data.torques))
            dz_values.append(dz)
            
        torques_all = np.array(torques_all)
        dz_values_all = np.array(dz_values)
        self.get_logger().info(f"Plotting torque values...")
        plt.figure(figsize=(10, 6))
        for i in range(6):  # Plot each joint separately
            plt.plot(dz_values_all, torques_all[:, i], '.-', label=f'Joint {i+1}')
        plt.xlabel('Z Position (mm)')
        plt.ylabel('Torque (%)')
        plt.legend()  
        plt.grid(True)
        plt.show()
        return dz_values_all, torques_all


    def place_chip_at_fixed_location(self, 
                                   robot_namespace: str, 
                                   # This pose is where the TCP should be when the gripper opens
                                   chip_drop_pose_params: tuple, # (x,y,z_at_drop, a,b,g)
                                   approach_offset_z_mm: float = 3.5, # How far above drop_z to approach first
                                   gripper_open_pos_mm: float = 14.0,
                                   retract_after_drop_mm: float = 10.0,
                                   # Add other parameters like velocity if needed
                                   placement_joint_vel_percent: float = 5.0
                                   ) -> bool:
        """
        Places a held chip at a specified location by approaching, moving to a drop height,
        opening the gripper, and retracting.

        Args:
            chip_drop_pose_params: Tuple (x,y,z,a,b,g) where the TCP should be
                                   (e.g., 1mm above surface) when opening the gripper.
            approach_offset_z_mm: Initial approach height above the final drop_z.
        Returns:
            bool: True if chip placement sequence presumed successful, False otherwise.
        """
        self.get_logger().info("--- Starting Chip Placement Sequence ---")
        drop_x, drop_y, drop_z, drop_a, drop_b, drop_g = chip_drop_pose_params

        # --- 1. Preparation ---
        # Assume chip is already held (gripper closed appropriately)
        self.set_blending(robot_namespace, 0) # For precise positioning

        # --- 2. Approach Above Destination ---
        self.get_logger().info(f"Moving to approach pose: Z={drop_z:.2f}")
        if not self.move_pose_and_wait(robot_namespace, drop_x, drop_y, drop_z, 
                                     drop_a, drop_b, drop_g, 
                                     timeout_s=15.0, pos_tol_mm=1.0, orient_tol_deg=1.0):
            self.get_logger().error("Failed to reach approach pose above destination.")
            return False
        time.sleep(0.2); rclpy.spin_once(self) # Settle

        # --- 3. Move to Final Drop Height ---
        self.set_joint_vel(robot_namespace, placement_joint_vel_percent) 
        self.get_logger().info(f"Moving to final drop height")
        if not self.move_lin_rel_wrf(robot_namespace, 0.0, 0.0, -approach_offset_z_mm, 0.0, 0.0, 0.0):
            self.get_logger().error("Failed to move to final drop height.")
            return False
        time.sleep(0.5); # Settle at drop height before opening gripper

        # --- 4. Open Gripper (Release Chip) ---
        self.get_logger().info(f"Opening gripper to {gripper_open_pos_mm}mm to release chip.")
        if not self.move_gripper(robot_namespace, position=gripper_open_pos_mm): # Ensure this blocks or add appropriate wait
             self.get_logger().error("Failed to open gripper.")
             return False
        time.sleep(0.5)  # Allow gripper to physically open fully and chip to drop

        # --- 5. Retract Upwards ---
        self.get_logger().info(f"Retracting upwards by {retract_after_drop_mm}mm.")
        # Assuming self.move_lin_rel_wrf is blocking or you use a _and_wait version
        if not self.move_lin_rel_wrf(robot_namespace, 0.0, 0.0, retract_after_drop_mm, 0.0, 0.0, 0.0):
             self.get_logger().error("Failed to retract after drop.")
             # Since the chip was successfully dropped, we consider this a success
             # but log the retraction failure for monitoring
             self.get_logger().warn("Chip was placed but robot failed to retract properly.")
             return True
        else:
            time.sleep(0.2); rclpy.spin_once(self) # Settle after retract

        self.get_logger().info("--- Chip Placement Sequence Presumed Successful ---")
        return True

  
            
'''
Run this after you have started up the meca_driver node and are connected to the robot (and have started the motion planner):
ros2 run meca_controller meca_control
'''
def main(args=None):
    rclpy.init(args=args) # init ros2 communications

    node = Meca_Control() # create node
    
    try:
        node.wait_for_initialization() # wait for subscribers to receive critical data (current joint angles)
        node.run() # run user code
        rclpy.spin(node) # continues to run node indefinitely until cancel with CTRL-C
    except KeyboardInterrupt:
        node.destroy_node()
        rclpy.try_shutdown() # "Shutdown this context, if not already shutdown." shuts down ros2 communications and nodes.
