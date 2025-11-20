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

import math
import time
import traceback
import serial

import numpy as np

import rclpy # python library for ros2. needed for every ros2 node
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

import tf2_ros
import tf2_geometry_msgs
from scipy.spatial.transform import Rotation as R

from geometry_msgs.msg import Pose, PoseStamped, Quaternion, TransformStamped, Vector3, Twist, Accel
from sensor_msgs.msg import JointState
from std_srvs.srv import Trigger
from std_msgs.msg import Header
from builtin_interfaces.msg import Duration as RosDurationMsg

from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import MotionPlanRequest, WorkspaceParameters, RobotState, Constraints, JointConstraint, PlanningOptions, PositionIKRequest
from moveit_msgs.srv import GetPositionIK 

from custom_interfaces.srv import GetMotionPlan, VisualizeMotionPlan, MoveJoints
from custom_interfaces.srv import GoToPose, MoveGripper, SetBlending, SetGripperForce, SetGripperVel, SetJointVel, SetJointAcc, MoveLin, MoveLinRelWrf, MovePoseEuler, WaitIdle, GetRtJointTorq, ClearMotion, Home
from custom_interfaces.msg import RobotStatus, GripperStatus
from custom_interfaces.msg import PoseStampedArray
from custom_interfaces.msg import CartesianTrajectory, CartesianTrajectoryPoint, CartesianPosture, CartesianTolerance
from custom_interfaces.action import FollowCartesianTrajectory

from meca_controller.meca_settings import ROBOT1 # ADJUST THESE GLOBAL CONSTANTS

# never use scientific notation, always 3 digits after the decimal point
np.set_printoptions(suppress=True, precision=3) 

class Meca_Control(Node):
    def __init__(self):
        super().__init__("meca_control")

        self._robot1_ns = ROBOT1['namespace']
        self._move_speed = ROBOT1['move_speed']

        # Subscribe to keep up to date on current joint angles, pose, and status of the robot (may not need all these):
        # We always just want the latest information.
        self.joint_subscriber_robot1_ = self.create_subscription(JointState, "/joint_states",
                                                                 self.update_robot1_joints_callback, 1)
        self.pose_subscriber_robot1_ = self.create_subscription(Pose, f"{self._robot1_ns}/pose",
                                                                 self.update_robot1_pose_callback, 1)
        self.gripper_subscriber_robot1_ = self.create_subscription(GripperStatus, f"{self._robot1_ns}/gripper_status",
                                                                 self.update_robot1_gripper_status_callback, 1)
        self.status_subscriber_robot1_ = self.create_subscription(RobotStatus, f"{self._robot1_ns}/robot_status",
                                                                 self.update_robot1_status_callback, 1)

        # Initialize variables that will be set later:
        self.current_joints_robot1 = None
        self.current_pose_robot1 = None
        self.current_gripper_status_robot1 = None
        self.current_robot1_status = None

        # Define the QoS profile for sensor data (latest data only)
        qos_profile_sensor_data = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=DurabilityPolicy.VOLATILE
        )

        self.chip_poses_subscriber_ = self.create_subscription(
            PoseStampedArray,
            '/chip_pose_calculator/detected_chip_world_poses',
            self.chip_poses_callback,
            qos_profile=qos_profile_sensor_data)  # Use explicit QoS profile
        self.latest_chip_poses_ = [] # List to store all received chip poses
        self.get_logger().info("Subscribed to /chip_pose_calculator/detected_chip_world_poses with sensor QoS")
        
        # Add refresh detection service client
        self.refresh_detection_client = self.create_client(
            Trigger, 
            '/chip_pose_calculator/refresh_detection'
        )
        self.get_logger().info("Created refresh detection service client")
        
        # Add debugging for topic connection
        self.get_logger().info("=== TOPIC SUBSCRIPTION DEBUG ===")
        self.get_logger().info(f"Topic: /chip_pose_calculator/detected_chip_world_poses")
        self.get_logger().info(f"QoS Profile: RELIABLE, KEEP_LAST, DEPTH=1, VOLATILE")
        self.get_logger().info(f"Message type: PoseStampedArray")
        self.get_logger().info("=== END SUBSCRIPTION DEBUG ===")
        
        # Add MoveIt2 interface
        self._moveit_callback_group = ReentrantCallbackGroup()
        
        # Action client for MoveIt2
        self.moveit_action_client = ActionClient(
            self,
            MoveGroup,
            'move_action',
            callback_group=self._moveit_callback_group
        )
        
        # Wait for MoveIt to be available
        if not self.moveit_action_client.wait_for_server(timeout_sec=2.0):
            self.get_logger().warn('MoveIt action server not available, proceeding without it.')
            self.moveit_available = False
        else:
            self.get_logger().info('MoveIt action server found and ready.')
            self.moveit_available = True

        self._cart_traj_action_client = ActionClient(self, 
                                                     FollowCartesianTrajectory, 
                                                     f'{self._robot1_ns}/meca_arm_controller/follow_cartesian_trajectory')
       
        # Wait for Cartesian Trajectory to be available
        if not self._cart_traj_action_client.wait_for_server(timeout_sec=2.0):
            self.get_logger().error('Cartesian Trajectory action server not available. Needed for some pathplanning.')
            raise Exception('Cartesian Trajectory action server not available. Needed for some pathplanning.')

        # TF buffer for coordinate transformations
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Add IK Service Client
        self.ik_client = self.create_client(GetPositionIK, '/compute_ik')
        
        self.meca_arm_joint_names = [
            "meca_axis_1_joint", "meca_axis_2_joint", "meca_axis_3_joint",
            "meca_axis_4_joint", "meca_axis_5_joint", "meca_axis_6_joint"
        ]

        self.wait_idle_client = self.create_client(WaitIdle, f"{self._robot1_ns}/wait_idle")
        self.clear_motion_client = self.create_client(ClearMotion, f"{self._robot1_ns}/clear_motion")
        self.home_client = self.create_client(Home, f"{self._robot1_ns}/home")

        self.move_pose_client = self.create_client(MovePoseEuler, f'{self._robot1_ns}/move_pose') 

        # --- Simplified Chip Detection ---
        # The detection node now provides stable poses, so we just store them directly
        self.latest_chip_poses_ = []  # List to store stable chip poses
        
        # --- Z Height Measurement Storage ---
        self.measured_grasp_z_height_ = None  # Store measured Z height for reuse
        
        # --- Gripper Configuration ---
        self.GRIPPER_OPEN_POS_MM = 18.0      # Gripper position when fully open
        self.GRIPPER_CLOSED_POS_MM = 9.5     # Gripper position when closed on chip
        self.GRIPPER_PLACE_POS_MM = 14.0     # Gripper position when placing chip (partially open)

        # --- Message Tracking ---
        self.message_counter = 0
        self.last_message_time = None

    def wait_for_initialization(self, timeout_length=2):
        """
        Purpose: Parts of this code rely upon getting information from topics published to by the meca_driver, such as current_joints.
                Errors will occur if the main code is executed before these are assigned (e.g. motion plan but not yet received the
                current joint states from the subscriber). This function will wait until the critical variables are set so that all
                the code runs as intended.

                -timeout_length: amount of time in seconds that should wait to receive data from both robots before beginning code execution;
                            after timeout period, just waits for data from one robot to begin because assumes you are only using 1 robot.

                TODO add to the critical init variables as code is built.
        """
        time_start = time.time()
        print('waiting to receive meca state data...')
        
        while rclpy.ok():
            rclpy.spin_once(self) # spin once while we wait to avoid blocking any callbacks. (like the data we are waiting on)
            if (self.current_joints_robot1 is not None):
                print('...received data for robot, starting now.')
                return

            # If timeout period has elapsed, only check for data being received from one robot before starting:
            if ((time.time() - time_start) >= timeout_length) and (self.current_joints_robot1 is not None):
                print('...received data for at least one robot, starting now because timeout has occurred waiting.\n')
                return

    def run(self):
        """
        Purpose: A space to write user code to control the robots. Example commands given here.
        
        TODO in the future, create a better way for the user to add code that doesn't involve editing this file.
        Currently unsure how to do this.

        NOTE: Everything involving the ros2 collision detection / urdf / visualization / motion planning has joint
        angles in RADIANS. The mecademicpy interface uses DEGREES for the MoveJoints command.
        """
        # Safety config
        self.set_joint_vel(self._robot1_ns, self._move_speed)
        self.set_gripper_force(self._robot1_ns, 5)
        self.move_gripper(self._robot1_ns, position=self.GRIPPER_OPEN_POS_MM)

        self.move_joints(np.array([0, 0, 0, 0, 0, 0]), self._robot1_ns)
        self.home(self._robot1_ns)

        pose = [180.0, 110.0, 136.577, 0.0, 90.0, 0.0]
        #pose = [190.00001, 0, 141.07091, 0, 90, 0]
        pose_reached = self.move_pose_and_wait(self._robot1_ns, pose[0], pose[1], pose[2], pose[3], pose[4], pose[5])
        if pose_reached:
            self.get_logger().info("Pose reached successfully.")
        else:
            self.get_logger().error("Failed to reach pose.")
        time.sleep(3.0) # wait for the chip detection to become active
        
        # Trigger refresh after robot moves to new position
        self.get_logger().info("Triggering chip detection refresh after robot movement...")
        refresh_success = self.refresh_chip_detection(wait_for_completion=True, timeout_seconds=10)
        if refresh_success:
            self.get_logger().info("Chip detection refresh completed successfully")
        else:
            self.get_logger().warn("Chip detection refresh may have failed, continuing anyway")
        
        input("Press Enter to continue...")
        
        # Check if the topic is publishing
        self.get_logger().info("=== TOPIC AVAILABILITY CHECK ===")
        topic_info = self.get_topic_names_and_types()
        chip_topic_found = False
        for topic_name, topic_types in topic_info:
            if '/chip_pose_calculator/detected_chip_world_poses' in topic_name:
                chip_topic_found = True
                self.get_logger().info(f"Found chip topic: {topic_name} with types: {topic_types}")
                break
        
        if not chip_topic_found:
            self.get_logger().warn("Chip topic not found in available topics!")
            self.get_logger().info("Available topics:")
            for topic_name, topic_types in topic_info:
                if 'chip' in topic_name.lower() or 'pose' in topic_name.lower():
                    self.get_logger().info(f"  {topic_name}: {topic_types}")
        else:
            self.get_logger().info("Chip topic is available")
        self.get_logger().info("=== END TOPIC CHECK ===")
        
        self.get_logger().info("Waiting for stable chip poses from detection node...")
        
        # Wait for stable chip poses from the detection node
        max_wait_time = 10.0  # Maximum time to wait for detections
        start_time = time.time()
        
        while rclpy.ok() and (time.time() - start_time) < max_wait_time:
            rclpy.spin_once(self, timeout_sec=0.1)
            time.sleep(0.1)
            
            if len(self.latest_chip_poses_) > 0:
                self.get_logger().info(f"Received {len(self.latest_chip_poses_)} stable chip poses from detection node")
                break
            else:
                self.get_logger().info("Waiting for stable chip poses...")

        if not rclpy.ok():
            self.get_logger().error("Node shutting down.")
            return
            
        if len(self.latest_chip_poses_) == 0:
            self.get_logger().error(f"Timeout waiting for chip detections. No stable poses received.")
            return
        
        # Use the stable poses directly from the detection node
        robust_chip_poses = self.latest_chip_poses_
        
        self.get_logger().info(f"Found {len(robust_chip_poses)} stable chip(s) to process.")
        for i, pose in enumerate(robust_chip_poses):
            self.get_logger().info(f"  Chip {i+1}: P({pose.pose.position.x*1000:.1f}, {pose.pose.position.y*1000:.1f}, {pose.pose.position.z*1000:.1f})mm")

        # Measure Z height only once for the first chip
        measured_grasp_z_height = None
        
        # Process each robust chip
        for i, chip_pose_msg in enumerate(robust_chip_poses):
            self.get_logger().info(f"Processing chip {i+1}/{len(robust_chip_poses)}")
            
            # Configuration for stirring
            circle_radius_mm = 8.0  # 8mm radius
            circle_points = 16      # 16 points 
            overhead_seconds = 5.5  # Overhead for function call + move up
            move_z_distance_mm = 60.0  # Distance to move up (180.0 - 121.8 = 58.2mm)
            beaker_height_mm = 121.8  # Height of beakers
            safe_height_mm = beaker_height_mm + move_z_distance_mm  # Safe height for moving between beakers
            self.get_logger().info(f"Chip pose: {chip_pose_msg}")
            # Transform chip pose to TCP target for grasping
            grasp_target = self.transform_chip_pose_to_tcp_target(chip_pose_msg, target_z_mm=95.0)
            if grasp_target is None:
                self.get_logger().error(f"Failed to transform chip pose for grasping.")
                continue
            self.get_logger().info(f"Grasp target: {grasp_target}")
            # Call the goto_chip_pose function with the pre-transformed target
            #input("Press Enter to continue...")
            success = self.goto_chip_pose(tcp_target=grasp_target)
            if success:
                self.get_logger().info("Successfully moved to chip pose.")
                # Now grasp the chip from the current position
                if i == 0:
                    # First chip: measure Z height with full surface detection
                    grasp_success = self.grasp_chip_fixed_location_by_lifting(self._robot1_ns)
                    if grasp_success:
                        measured_grasp_z_height = self.get_measured_grasp_z_height()
                        self.get_logger().info(f"Measured grasp Z height: {measured_grasp_z_height:.2f}mm")
                    else:
                        self.get_logger().error("Failed to grasp the first chip.")
                        continue  # Skip to next chip if grasp failed
                else:
                    # Subsequent chips: use measured Z height for fast approach
                    if measured_grasp_z_height is not None:
                        grasp_success = self.grasp_chip_fast_approach(self._robot1_ns, measured_grasp_z_height)
                    else:
                        self.get_logger().error("No measured Z height available for subsequent chip.")
                        continue
                
                self.set_joint_vel(self._robot1_ns, self._move_speed)
                if grasp_success:
                    self.get_logger().info("Successfully grasped the chip.")
                else:
                    self.get_logger().error("Failed to grasp the chip.")
                    continue  # Skip to next chip if grasp failed
            else:
                self.get_logger().error("Failed to move to chip pose.")
                continue  # Skip to next chip if movement failed

            # Only proceed if grasp_success is True
            self.set_joint_vel(self._robot1_ns, self._move_speed)
            self.move_joints(np.array([0, 0, 0, 0, 0, 0]), self._robot1_ns)
            self.move_gripper(self._robot1_ns, position=8.5)

            # --- Stir in first beaker ---
            first_beaker_success = self.stir_in_beaker(
                beaker_x=193.0, beaker_y=18.5, beaker_z=beaker_height_mm,
                stir_duration_seconds=32.5, stir_frequency=1, 
                circle_radius_mm=circle_radius_mm,
                circle_points=circle_points,
                move_z_distance_mm=move_z_distance_mm, move_z_distance_time = 0.2,
                beaker_name="first beaker",
                move_speed=self._move_speed
            )

            # --- Go to second beaker ---
            self.move_pose_and_wait(self._robot1_ns, 193.0, -78.0, safe_height_mm, 0, 90, 0)
            self.move_pose_and_wait(self._robot1_ns, 193.0, -78.0, beaker_height_mm, 0, 90, 0)

            # --- Stir in second beaker ---
            second_beaker_success = self.stir_in_beaker(
                beaker_x=193.0, beaker_y=-78.0, beaker_z=beaker_height_mm,
                stir_duration_seconds=12.0, stir_frequency=1, 
                circle_radius_mm=circle_radius_mm,
                circle_points=circle_points,
                move_z_distance_mm=move_z_distance_mm, move_z_distance_time = 0.2,
                beaker_name="second beaker",
                move_speed=self._move_speed
            )
            self.get_logger().info("Stirring in second beaker complete.")
            self.set_joint_vel(self._robot1_ns, self._move_speed)
            if not second_beaker_success:
                self.get_logger().error("Failed to stir in second beaker.")
                self.move_pose_and_wait(self._robot1_ns, 193.0, -78.0, safe_height_mm, 0, 90, 0)
            #self.move_pose_and_wait(self._robot1_ns, 193.0, -78.0, 200.0, 0, 90, 0)
            # go to N2 gun

            #pose_N2 = [25.8, -204.5, 84.7, 75.6, -13.5, -100.5]
            pose_N2 = [25.8, -204.5, 83.5, 75.6, -13.5, -96.0]
            pose_reached = self.move_pose_and_wait(self._robot1_ns, pose_N2[0], pose_N2[1], pose_N2[2], pose_N2[3], pose_N2[4], pose_N2[5])
            if pose_reached:
                self.get_logger().info("N2 pose reached successfully.")
            else:
                self.get_logger().error("Failed to reach N2 pose.")

            self.switch_usb_relay(delay_on_s=5.0)
            self.move_joints(np.array([0, 0, 0, 0, 0, 0]), self._robot1_ns)
            self.home(self._robot1_ns)


            # Switch USB relay before going to N2 gun
            # Use the same grasp_target for placement (same location, different height)
            # The place_chip_at_fixed_location function will handle the Z offset internally
            placement_success = self.place_chip_at_fixed_location(
                robot_namespace=self._robot1_ns,
                chip_drop_pose_params=grasp_target)
            if placement_success:
                self.get_logger().info("Chip placement sequence successful - chip placed back at original location.")
            else:
                self.get_logger().error("Chip placement sequence failed.")
                
            self.set_joint_vel(self._robot1_ns, self._move_speed)
            

                

        #self.touchdown_torque_plotter(self._robot1_ns,approach_pose_params=(190.0, 158.943, 96.577, 0.0, 90.0, 0.0), step_mm=0.1, timeout_s=120.0, N=60)
        self.set_joint_vel(self._robot1_ns, self._move_speed)

        #input("Press Enter here to move back to chip view...")
        pose = [180.0, 110.0, 136.577, 0.0, 90.0, 0.0]
        #pose = [190.00001, 0, 141.07091, 0, 90, 0]
        pose_reached = self.move_pose_and_wait(self._robot1_ns, pose[0], pose[1], pose[2], pose[3], pose[4], pose[5])
        if pose_reached:
            self.get_logger().info("Pose reached successfully.")
        else:
            self.get_logger().error("Failed to reach pose.")
        
    def update_robot1_joints_callback(self, joints):
        self.current_joints_robot1 = joints # JointState (has .position and .velocity fields)

    def update_robot1_pose_callback(self, pose_msg):
        self.current_pose_robot1 = pose_msg # Pose (has .position and .orientation fields)
    
    def update_robot1_gripper_status_callback(self, gripper_status):
        self.current_gripper_status_robot1 = gripper_status # custom msg type

    def update_robot1_status_callback(self, robot_status):
        self.current_robot1_status = robot_status # custom msg type



    def determine_which_robot_from_namespace(self, namespace):
        """
        Purpose: used to determine from the namespace whether this is robot1 or robot2, for the purpose of getting the order right
                in the len 12 joint angle array sent to the motion planner and visualization.
        """
        is_robot1 = True
        return is_robot1


    def get_current_joints_both_robots(self, required):
        """
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

        """
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
    
    def move_joints(self, desired_joint_angles, robot_namespace, error_tolerance=.1, timeout_length=60):
        """
        Service call

        Inputs:
        - desired_joint_angles (degrees): np.array of len 6.
        - robot_namespace: the namespace of the robot which you want to control. Should be /robot1 or /robot2
        - error_tolerance: (degrees) the joint angle tolerance within which the robot should reach before executing the next command
        - timeout_length: the max amount of time in seconds to wait on the robot to reach the desired position before returning failure

        Returns:
        - is_reached: boolean, whether the robot successfully reached the desired position.
        """
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
    
    def move_gripper(self, robot_namespace, position, command="pos"):
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
        client = self.move_pose_client 
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
        return future # Assume sent if no immediate exception
        
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
    

    def has_reached_config(self, desired_joint_angles, current_joint_angles, error_tolerance=.1):
        """
        Purpose: Used to see whether the robot has reached its desired configuration, to a tolerance (in degrees).
        """
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




    def set_blending(self, robot_namespace, blending):
        """Set Blending 
        Service call

        Request inputs:
            - blending [float] from 0 to 100
        
        Response:
            - error [bool]: True if error occurred, False otherwise.
        """
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
        client = self.create_client(MoveJoints, f"{self._robot1_ns}/move_joints")
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

    def set_gripper_force(self, robot_namespace, gripper_force):
        """ Set Force of Gripper
            Request inputs:
                - gripper_force: from 5 to 100, which is a percentage of the maximum force the MEGP 25E gripper can hold (40N).
            
            Response:
                - error [bool]: True if error occurred, False otherwise.
        """
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

    def set_gripper_vel(self, robot_namespace, gripper_vel):
        """
        Service call

        Request inputs:
            - gripper_vel: from 5 to 100, which is a percentage of the maximum finger velocity of the MEGP 25E gripper (∼100 mm/s).
        
        Response:
            - error [bool]: True if error occurred, False otherwise.
        """
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
        
    def set_joint_vel(self, robot_namespace, joint_vel):
        """
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
        """
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


    def set_joint_acc(self, robot_namespace, joint_acc):
        """
        Service call

        Request inputs:
            - joint_acc: from 0.001 to 150, which is a percentage of maximum acceleration of the joints, ranging from 0.001% to 150%
        Response:
            - error [bool]: True if error occurred, False otherwise.
        """
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
   
    def move_circular_time_freq(self, radius_mm: float, num_points: int, duration_seconds: float, frequency: float,
                                robot_namespace: str, clockwise: bool = True, 
                                move_z_distance_mm: float = 20.0, move_z_time_seconds: float = 0.2):
        """
        Moves the robot TCP in circles for a specified time duration with a given frequency.
        
        This method generates a complete Cartesian trajectory with timing information and submits
        it to the robot controller via the FollowCartesianTrajectory action client. The trajectory
        includes all waypoints that fit within the specified duration at the given frequency.
        
        Args:
            radius_mm (float): The radius of the circle in mm.
            num_points (int): Number of waypoints to approximate each complete circle lap.
            duration_seconds (float): Total duration for circular motion in seconds.
            frequency (float): Number of laps per second (must be between 0.1 and 3.0).
            robot_namespace (str): Namespace of the target robot (e.g., '/robot1').
            clockwise (bool): Direction of the circle motion. True for clockwise, False for counter-clockwise.
            move_z_distance_mm (float): Distance to move up/down before/after completing circular motion. Default 20.0mm.
            move_z_time_seconds (float): Time to cover the distance. Default 0.2 s
            
        Returns:
            bool: True if the circular motion completed successfully, False otherwise.
        """
        # Input validation for frequency
        if not (0.1 <= frequency <= 3.0):
            self.get_logger().error(f"Frequency must be between 0.1 and 3.0, got {frequency}")
            return False
        
        if duration_seconds <= 0.5:
            self.get_logger().error("Duration must be greater than 0.5 seconds.")
            return False

        overall_success = False

        self.get_logger().info(f"Starting timed circular move for {robot_namespace}: Radius={radius_mm} mm, Points={num_points}, Duration={duration_seconds}s, Frequency={frequency} Hz")

        if num_points < 3:
            self.get_logger().error("Number of points for circle must be at least 3.")
            return False

        # --- Get the starting/center pose ---
        start_pose = self.current_pose_robot1
        if start_pose is None:
            self.get_logger().error("Cannot execute circular move: No current pose feedback received.")
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
        
        self.get_logger().info(f"Circle center based on current pose: "
                               f"X={cx:.1f}, Y={cy:.1f}, Z={cz:.1f} / (-{move_z_distance_mm:.1f}), "
                               f"A={ca:.1f}, B={cb:.1f}, G={cg:.1f}")
            
        # Calculate total number of waypoints based on frequency
        circle_waypoints = int(frequency * duration_seconds * num_points)
        if circle_waypoints < num_points:
            self.get_logger().info(
                f"Frequency {frequency} Hz and duration {duration_seconds}s only allow {circle_waypoints} waypoints, "
                f"but {num_points} points per circle requested. The circle will not be complete, which is expected for this duration."
            )

        self.get_logger().info(f"Generating {circle_waypoints} waypoints for {duration_seconds}s duration at {frequency} Hz frequency")
        
        # --- Generate all waypoints that fit into the time duration ---
        waypoints = []
        angle_step = 2 * math.pi / num_points
        
        for i in range(circle_waypoints): 
            # Calculate which circle we're on and the angle within that circle
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

        # --- Create Cartesian Trajectory with timing information ---
        self.get_logger().info(f"Creating Cartesian trajectory with {len(waypoints)} waypoints.")
        
        # Create the trajectory goal
        goal = FollowCartesianTrajectory.Goal()
        
        # Set up the trajectory
        trajectory = CartesianTrajectory()
        trajectory.header.frame_id = "meca_base_link"  # Assuming this is the correct frame
        trajectory.header.stamp = self.get_clock().now().to_msg()
        trajectory.controlled_frame = "meca_tcp_link"  # Assuming this is the controlled frame
        
        time_start_circle = 0
        if move_z_distance_mm > 0: # need to start with moving down.
            time_start_circle = move_z_time_seconds # Give some time to move to first waypoint

        # Calculate time step between waypoints for the circle based on frequency
        time_step_seconds = duration_seconds / circle_waypoints
        
        # Create trajectory points with timing
        for i, wp in enumerate(waypoints):
            x, y, z, a, b, g = wp
            
            # Create trajectory point
            point = CartesianTrajectoryPoint()
            
            # Set timing (time from start of trajectory)
            time_from_start = time_start_circle + i * time_step_seconds
            point.time_from_start.sec = int(time_from_start)
            point.time_from_start.nanosec = int((time_from_start - int(time_from_start)) * 1e9)
            
            # Create pose (convert from Euler angles to quaternion)
            pose = Pose()
            pose.position.x = x
            pose.position.y = y
            pose.position.z = z
            
            # Convert Euler angles (degrees) to quaternion
            # Note: This assumes the angles are in the order A, B, G (roll, pitch, yaw)
            quat = R.from_euler('xyz', [a, b, g], degrees=True).as_quat()
            pose.orientation.x = quat[0]
            pose.orientation.y = quat[1]
            pose.orientation.z = quat[2]
            pose.orientation.w = quat[3]
            point.pose = pose 
            trajectory.points.append(point)
        
        goal.trajectory = trajectory
        
        # Tolerances not used.
        goal.path_tolerance = CartesianTolerance()       
        goal.goal_tolerance = CartesianTolerance()
        
        # Set goal time tolerance
        goal.goal_time_tolerance.sec = 1
        goal.goal_time_tolerance.nanosec = 0
        
        if move_z_distance_mm > 0:
            # Return to lifted position
            point = CartesianTrajectoryPoint()

            time_from_start = time_start_circle + (len(waypoints)-1) * time_step_seconds + move_z_time_seconds
            point.time_from_start.sec = int(time_from_start)
            point.time_from_start.nanosec = int((time_from_start - int(time_from_start)) * 1e9)

            pose = Pose()
            pose.position.x = cx
            pose.position.y = cy
            pose.position.z = cz_raised

            quat = R.from_euler('xyz', [ca, cb, cg], degrees=True).as_quat()
            pose.orientation.x = quat[0]
            pose.orientation.y = quat[1]
            pose.orientation.z = quat[2]
            pose.orientation.w = quat[3]
            point.pose = pose
            trajectory.points.append(point)
        
        # --- Submit trajectory to action client ---
        self.get_logger().info(f"Submitting Cartesian trajectory with {len(trajectory.points)} total points / ({len(waypoints)} circle waypoints)")
        start_time = time.time()
        # Send goal and wait for result
        send_goal_future = self._cart_traj_action_client.send_goal_async(goal)
        
        # Wait for goal to be accepted
        rclpy.spin_until_future_complete(self, send_goal_future, timeout_sec=10.0)
        if not send_goal_future.done():
            self.get_logger().error("Failed to send trajectory goal within timeout")
            overall_success = False
        else:
            goal_handle = send_goal_future.result()
            if not goal_handle.accepted:
                self.get_logger().error("Trajectory goal was rejected")
                overall_success = False
            else:
                self.get_logger().info("Trajectory goal accepted, waiting for completion...")
                
                # Wait for result
                get_result_future = goal_handle.get_result_async()
                rclpy.spin_until_future_complete(self, get_result_future, timeout_sec=duration_seconds + 10.0)
                
                if not get_result_future.done():
                    self.get_logger().error("Trajectory execution timed out")
                    overall_success = False
                else:
                    result = get_result_future.result().result
                    
                    # Check result
                    if result.error_code == FollowCartesianTrajectory.Result.SUCCESSFUL:
                        self.get_logger().info("Cartesian trajectory completed successfully")
                        overall_success = True
                    else:
                        self.get_logger().error(f"Cartesian trajectory failed with error code {result.error_code}: {result.error_string}")
                        overall_success = False

        self.wait_idle(robot_namespace, timeout_s=1)
        final_elapsed = time.time() - start_time

        if overall_success:
            self.get_logger().info("Timed circular motion complete. Included moving up from current position.")          
        else:
            # If motion failed, still try to move up from current position
            self.get_logger().warn("Circular motion failed, attempting to move up from current position...")
            self.wait_idle(robot_namespace, timeout_s=1)

            if not self.move_lin_rel_wrf(robot_namespace, 0.0, 0.0, move_z_distance_mm, 0.0, 0.0, 0.0):
                self.get_logger().error("Failed to move up after circle failure.")
            else:
                self.get_logger().info("Moved up after failure.")

        self.get_logger().info(f"Timed circular motion finished. Total time in controller: {final_elapsed:.1f}s, Waypoints in trajectory: {len(waypoints)}, Success: {overall_success}")
        return overall_success

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
        initial_press_depth_mm: float = 5.0, # Press down this much past approach to ensure contact
        lift_step_mm: float = 0.1,          # Small steps for lifting to find release
        max_lift_search_mm: float = 3.0,    
        # --- Torque parameters based on your plot (VALUES ARE PERCENTAGES as per plot) ---
        # We'll use Joint 5 as the primary indicator.
        # When pressed, J5 torque is high (e.g., > 25%). When free, it's low (e.g., < 10% and often negative).
        contact_joint_indices: list[int] = [4], # Focus on Joint 5 (index 4)
        torque_delta_release_threshold_percent: float = 0.8, # Small change threshold - indicates we're in free-air
        press_torque_increase_threshold_percent: float = 10.0, # Example: J5 must increase by at least 10% from free-air
        significant_drop_threshold_percent: float = 15.0, # Large drop threshold - indicates we've started releasing
        # ---------------------------------------------------------------------------------
        grasp_height_above_surface_mm: float = 0.2, # For a 0.5mm thick chip
        gripper_open_pos_mm: float = None,  # Use centralized config if None
        gripper_closed_pos_mm: float = None, # Use centralized config if None
        lift_chip_height_mm: float = 5.0
    ) -> bool:
        # Use centralized gripper configuration if not specified
        if gripper_open_pos_mm is None:
            gripper_open_pos_mm = self.GRIPPER_OPEN_POS_MM
        if gripper_closed_pos_mm is None:
            gripper_closed_pos_mm = self.GRIPPER_CLOSED_POS_MM
            
        self.get_logger().info("--- Starting Chip Grasp (Press Down, Lift to Detect Surface) ---")
        
        # Get current pose instead of using approach pose parameter
        if self.current_pose_robot1 is None:
            self.get_logger().error("No current pose available for grasp.")
            return False
            
        current_pose = self.current_pose_robot1
        approach_x = current_pose.position.x
        approach_y = current_pose.position.y
        approach_z_start = current_pose.position.z
        approach_a = current_pose.orientation.x
        approach_b = current_pose.orientation.y
        approach_g = current_pose.orientation.z

        # 1. Preparations
        if not self.move_gripper(robot_namespace, position=gripper_open_pos_mm): return False
        self.set_blending(robot_namespace, 0)

        # 2. We're already at the approach pose (from goto_chip_pose), so just settle
        self.get_logger().info(f"Already at approach pose Z: {approach_z_start:.2f}")
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
        self.get_logger().info(f"Lifting by {lift_step_mm}mm steps to detect surface release (large drop > {significant_drop_threshold_percent}% then small changes < {torque_delta_release_threshold_percent}%)...")
        surface_z_at_release = None
        previous_step_torques = np.copy(torques_at_full_press) # Torques when fully pressed
        pose_at_surface_contact = self.current_pose_robot1 # Pose when fully pressed (before starting to lift)
        if not pose_at_surface_contact: # Should have pose here
            self.get_logger().error("Critical: Lost pose feedback before starting lift detection."); return False

        max_lift_steps = int(max_lift_search_mm / lift_step_mm) + 1
        has_had_significant_drop = False  # Track if we've had a large drop from pressed state
        
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
                        # Check if we've had a significant drop from pressed state
                        torque_drop_from_pressed = torques_at_full_press[joint_idx] - current_torques_percent[joint_idx]
                        # Check if this step had a small change (indicating we're in free-air)
                        torque_change_this_step = abs(previous_step_torques[joint_idx] - current_torques_percent[joint_idx])
                        
                        self.get_logger().info(f"Lift step {step_num+1}, J5 Torque: Pressed={torques_at_full_press[joint_idx]:.2f}%, Curr={current_torques_percent[joint_idx]:.2f}%, Drop={torque_drop_from_pressed:.2f}%, StepChange={torque_change_this_step:.2f}%")
                        
                        # Check if we've had a significant drop from pressed state
                        if torque_drop_from_pressed > significant_drop_threshold_percent:
                            has_had_significant_drop = True
                            self.get_logger().info(f"Significant drop detected: {torque_drop_from_pressed:.2f}% > {significant_drop_threshold_percent}%")
                        
                        # Only look for small changes if we've already had a significant drop
                        if has_had_significant_drop and torque_change_this_step < torque_delta_release_threshold_percent:
                            self.get_logger().info(f"SURFACE RELEASE DETECTED on J5! Small change {torque_change_this_step:.2f}% < {torque_delta_release_threshold_percent}% after significant drop")
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
        
        # Store the measured Z height for reuse
        self.measured_grasp_z_height_ = target_grasp_z
        
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
                                   gripper_open_pos_mm: float = None, # Use centralized config if None
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
        # Use centralized gripper configuration if not specified
        if gripper_open_pos_mm is None:
            gripper_open_pos_mm = self.GRIPPER_PLACE_POS_MM
            
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
    
    def chip_poses_callback(self, msg):
        # Simply store the stable poses from the detection node
        self.latest_chip_poses_ = msg.poses
        #self.get_logger().info(f"Received {len(self.latest_chip_poses_)} chip poses from detection node")
    
    def refresh_chip_detection(self, wait_for_completion=True, timeout_seconds=10):
        try:
            # Wait for service to be available
            if not self.refresh_detection_client.wait_for_service(timeout_sec=5.0):
                self.get_logger().error("Refresh detection service not available")            
                return False
            
            # Call the refresh service
            request = Trigger.Request()
            future = self.refresh_detection_client.call_async(request)
            
            # Wait for service response
            rclpy.spin_until_future_complete(self, future, timeout_sec=5)
            
            if future.done():
                response = future.result()
                if response.success:
                    self.get_logger().info(f"Chip detection refresh triggered: {response.message}")
                    
                    if wait_for_completion:
                        # Wait for detection to complete (usually 2-3 seconds)
                        self.get_logger().info(f"Waiting {timeout_seconds} seconds for detection to complete...")
                        time.sleep(timeout_seconds)
                        
                        # Check if we received new poses
                        if self.latest_chip_poses_:
                            self.get_logger().info(f"Fresh detection complete! Got {len(self.latest_chip_poses_)} chip poses")
                            return True
                        else:
                            self.get_logger().warn("No chip poses received after refresh")
                            return False
                    else:
                        return True
                else:
                    self.get_logger().error(f"Refresh failed: {response.message}")
                    return False
            else:
                self.get_logger().error("Refresh service call timed out")            
                return False
                
        except Exception as e:
            self.get_logger().error(f"Error calling refresh service: {e}")
            return False

    def get_robust_chip_list(self, detection_history):
        """
        Simple grid-based robust chip detection.
        Takes frames until we have 7 frames with chips, then assigns chips to grid positions
        and averages the positions for each grid spot.
        """
        # First, ensure we have enough frames with chips
        frames_with_chips = [frame for frame in detection_history if len(frame) > 0]
        
        if len(frames_with_chips) < 7:
            self.get_logger().warn(f"Only {len(frames_with_chips)} frames have chips, need at least 7")
            if len(frames_with_chips) == 0:
                return []
        
        # Use the first 7 frames with chips
        frames_to_use = frames_with_chips[:7]
        self.get_logger().info(f"Using {len(frames_to_use)} frames with chips for robust detection")
        
        # Create a grid to assign chips to positions
        # We'll use a simple approach: sort chips by X coordinate, then by Y coordinate
        all_chip_positions = []
        
        # Collect all chip positions from all frames
        for frame_idx, frame in enumerate(frames_to_use):
            self.get_logger().info(f"Frame {frame_idx + 1}: {len(frame)} chips")
            for chip_idx, pose in enumerate(frame):
                pos = (pose.pose.position.x * 1000, pose.pose.position.y * 1000, pose.pose.position.z * 1000)
                all_chip_positions.append({
                    'frame': frame_idx,
                    'chip': chip_idx,
                    'pose': pose,
                    'pos': pos
                })
                self.get_logger().info(f"  Chip {chip_idx + 1}: P({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f})mm")
        
        # Group chips by grid position (simple clustering by proximity)
        grid_positions = self._cluster_chips_by_position(all_chip_positions)
        
        # Create robust poses from grid positions
        robust_poses = []
        for grid_idx, grid_pos in enumerate(grid_positions):
            if len(grid_pos['chips']) >= 2:  # At least 2 detections to be robust
                # Average the positions
                avg_x = sum(chip['pos'][0] for chip in grid_pos['chips']) / len(grid_pos['chips'])
                avg_y = sum(chip['pos'][1] for chip in grid_pos['chips']) / len(grid_pos['chips'])
                avg_z = sum(chip['pos'][2] for chip in grid_pos['chips']) / len(grid_pos['chips'])
                
                # Use the orientation from the first chip in this grid position
                first_pose = grid_pos['chips'][0]['pose']
                
                # Create averaged pose
                avg_pose = first_pose.__class__()
                avg_pose.header = first_pose.header
                avg_pose.pose.position.x = avg_x / 1000.0  # Convert back to meters
                avg_pose.pose.position.y = avg_y / 1000.0
                avg_pose.pose.position.z = avg_z / 1000.0
                avg_pose.pose.orientation = first_pose.pose.orientation
                
                robust_poses.append(avg_pose)
                
                self.get_logger().info(f"Grid position {grid_idx + 1}: {len(grid_pos['chips'])} detections, "
                                     f"avg pos: ({avg_x:.1f}, {avg_y:.1f}, {avg_z:.1f})mm")
        
        self.get_logger().info(f"Found {len(robust_poses)} robust chips using grid-based detection")
        return robust_poses
    
    def _cluster_chips_by_position(self, all_chip_positions, distance_threshold_mm=30.0):
        """
        Simple clustering of chips by position proximity.
        Returns list of grid positions, each containing chips that belong to that position.
        """
        grid_positions = []
        
        for chip_data in all_chip_positions:
            pos = chip_data['pos']
            assigned = False
            
            # Check if this chip belongs to an existing grid position
            for grid_pos in grid_positions:
                # Calculate distance to grid center
                grid_center = grid_pos['center']
                distance = ((pos[0] - grid_center[0])**2 + (pos[1] - grid_center[1])**2 + (pos[2] - grid_center[2])**2)**0.5
                
                if distance < distance_threshold_mm:
                    # Add to this grid position
                    grid_pos['chips'].append(chip_data)
                    # Update center
                    avg_x = sum(chip['pos'][0] for chip in grid_pos['chips']) / len(grid_pos['chips'])
                    avg_y = sum(chip['pos'][1] for chip in grid_pos['chips']) / len(grid_pos['chips'])
                    avg_z = sum(chip['pos'][2] for chip in grid_pos['chips']) / len(grid_pos['chips'])
                    grid_pos['center'] = (avg_x, avg_y, avg_z)
                    assigned = True
                    break
            
            if not assigned:
                # Create new grid position
                grid_positions.append({
                    'center': pos,
                    'chips': [chip_data]
                })
        
        return grid_positions

    def _pose_close(self, pose1, pose2, thresh):
        import numpy as np
        p1 = np.array([pose1.pose.position.x, pose1.pose.position.y, pose1.pose.position.z])
        p2 = np.array([pose2.pose.position.x, pose2.pose.position.y, pose2.pose.position.z])
        return np.linalg.norm(p1 - p2) < thresh

    def transform_chip_pose_to_tcp_target(self, chip_pose_msg, target_z_mm=95.0):
        """
        Transform a chip pose to a TCP target pose for the robot.
        
        Args:
            chip_pose_msg: The chip pose message containing position and orientation
            target_z_mm: Target Z height in mm (default 95.0)
        
        Returns:
            tuple: (x, y, z, alpha, beta, gamma) in mm and degrees, or None if transformation fails
        """
        # Get the chip's yaw for use in both calculations
        q_chip = chip_pose_msg.pose.orientation
        chip_yaw_deg = R.from_quat([q_chip.x, q_chip.y, q_chip.z, q_chip.w]).as_euler('xyz', degrees=True)[2]
        self.get_logger().info(f"Detected Chip Yaw: {chip_yaw_deg:.2f} degrees")
        
        # Get the static tool offset from the TF tree
        try:
            transform = self.tf_buffer.lookup_transform('meca_axis_6_link', 'tweezer_tcp', rclpy.time.Time())
            tcp_offset_local = np.array([
                transform.transform.translation.x,
                transform.transform.translation.y,
                transform.transform.translation.z
            ])
        except Exception as e:
            self.get_logger().error(f"Could not look up TF: {e}")
            return None

        # --- PART A: Calculate Flange Position using a temporary, simple orientation ---
        self.get_logger().info("--- Calculating Position ---")
        
        # To get the correct position, we use the simple, non-pitched orientation that worked in our tests.
        # This orientation points the tool straight down, aligned with the chip.
        calculation_orientation = R.from_euler('xyz', [180, 0, chip_yaw_deg+90.0], degrees=True)
        self.get_logger().info(f"Using temporary orientation for calculation: (180, 0, {chip_yaw_deg:.2f})")

        # Define the target position for the TCP
        P_tcp_target_m = np.array([
            chip_pose_msg.pose.position.x,
            chip_pose_msg.pose.position.y,
            target_z_mm / 1000.0  # Convert target Z from mm to meters
        ])

        # Calculate the flange position using the proven formula and our temporary "calculation" orientation
        tcp_offset_world = calculation_orientation.apply(tcp_offset_local)
        P_flange_target_m = P_tcp_target_m - tcp_offset_world
        
        self.get_logger().info(f"Calculated Flange Position (X, Y, Z meters): {P_flange_target_m}")

        # --- PART B: Assemble the Final Command using the calculated position but your MANUAL orientation ---
        self.get_logger().info("--- Assembling Final Command ---")
        # Define your manual, hardcoded orientation angles
        manual_alpha = 90.0
        manual_beta = 180.0+chip_yaw_deg
        manual_gamma = -90.0
        self.get_logger().info(f"Overriding with manual orientation: ({manual_alpha:.2f}, {manual_beta:.2f}, {manual_gamma:.2f})")

        # The final command uses the position from PART A and the orientation from PART B
        command_pose = [
            P_flange_target_m[0] * 1000.0,  # Use calculated X (in mm)
            P_flange_target_m[1] * 1000.0,  # Use calculated Y (in mm)
            target_z_mm,                    # Use provided Z height
            manual_alpha,                   # Use your manual Alpha
            manual_beta,                    # Use your manual Beta
            manual_gamma                    # Use your manual Gamma
        ]

        self.get_logger().info(f"Transformed chip pose to TCP target: "
                              f"P(mm): ({command_pose[0]:.2f}, {command_pose[1]:.2f}, {command_pose[2]:.2f}), "
                              f"O(XYZ deg): ({command_pose[3]:.2f}, {command_pose[4]:.2f}, {command_pose[5]:.2f})")
        
        return tuple(command_pose)

    def switch_usb_relay(self, delay_on_s=5):
        usb_relay = serial.Serial("/dev/ttyUSB0",9600)
        if usb_relay.is_open:
            self.get_logger().info(str(usb_relay))
            on_cmd = b'\xA0\x01\x01\xa2'
            off_cmd =  b'\xA0\x01\x00\xa1'


            usb_relay.write(on_cmd )
            time.sleep(delay_on_s)
            usb_relay.write(off_cmd)
            usb_relay.close()

        # if disconnecting and reconnecting the usb relay, run the following command
        # sudo chmod 666 /dev/ttyUSB0

    def goto_chip_pose(self, chip_pose_msg=None, target_z_mm=95.0, tcp_target=None):
        """
        Move the robot to a chip pose.
        
        Args:
            chip_pose_msg: The chip pose message containing position and orientation (optional if tcp_target provided)
            target_z_mm: Target Z height in mm (default 95.0, used only if tcp_target not provided)
            tcp_target: Pre-transformed TCP target pose tuple (x, y, z, alpha, beta, gamma) (optional if chip_pose_msg provided)
        
        Returns:
            bool: True if movement was successful, False otherwise
        """
        # Use provided tcp_target or transform chip_pose_msg
        if tcp_target is not None:
            x, y, z, alpha, beta, gamma = tcp_target
        else:
            if chip_pose_msg is None:
                self.get_logger().error("Either chip_pose_msg or tcp_target must be provided.")
                return False
            # Transform chip pose to TCP target
            tcp_target = self.transform_chip_pose_to_tcp_target(chip_pose_msg, target_z_mm)
            if tcp_target is None:
                self.get_logger().error("Failed to transform chip pose to TCP target.")
                return False
            x, y, z, alpha, beta, gamma = tcp_target
        
        current_p = self.current_pose_robot1
        self.get_logger().info(f"Current FLANGE Pose (mm, User Euler deg): P({current_p.position.x:.2f}, {current_p.position.y:.2f}, {current_p.position.z:.2f}), O({current_p.orientation.x:.2f}, {current_p.orientation.y:.2f}, {current_p.orientation.z:.2f}, {current_p.orientation.w:.2f})")
        #input("Press Enter to move the robot to this pose...")

        # Send the command
        if self.move_pose_and_wait(self._robot1_ns, x, y, z, alpha, beta, gamma):
            self.get_logger().info("TEST SUCCEEDED: Robot reports move is complete.")
            self.get_logger().info("--> Now, please check in RViz if the 'meca_axis_6_link' frame origin has overlapped with the chip pose marker origin.")
            return True
        else:
            self.get_logger().error("TEST FAILED: Robot could not reach the target pose.")
            return False

    def matrix_to_mecademic_euler_deg(self, matrix: np.ndarray) -> tuple[float, float, float]:
        """Converts a 4x4 matrix to XYZ Euler angles in degrees for the Meca500."""
        rotation_matrix = matrix[0:3, 0:3]
        # Use 'xyz' for mobile XYZ convention (Rx, Ry', Rz'')
        # This is CRUCIAL to match the robot's controller
        euler_rad = R.from_matrix(rotation_matrix).as_euler('xyz', degrees=False)
        
        alpha_deg = math.degrees(euler_rad[0]) # Roll
        beta_deg = math.degrees(euler_rad[1])  # Pitch
        gamma_deg = math.degrees(euler_rad[2]) # Yaw
        
        return alpha_deg, beta_deg, gamma_deg
    
    def get_measured_grasp_z_height(self):
        """Returns the measured grasp Z height that was stored during the first chip grasp."""
        return self.measured_grasp_z_height_
    
    def set_gripper_config(self, open_pos_mm: float = None, closed_pos_mm: float = None, place_pos_mm: float = None):
        """
        Update gripper configuration for different chip sizes.
        
        Args:
            open_pos_mm: Gripper position when fully open (default: keep current)
            closed_pos_mm: Gripper position when closed on chip (default: keep current)
            place_pos_mm: Gripper position when placing chip (default: keep current)
        """
        if open_pos_mm is not None:
            self.GRIPPER_OPEN_POS_MM = open_pos_mm
            self.get_logger().info(f"Updated gripper open position to {open_pos_mm}mm")
        if closed_pos_mm is not None:
            self.GRIPPER_CLOSED_POS_MM = closed_pos_mm
            self.get_logger().info(f"Updated gripper closed position to {closed_pos_mm}mm")
        if place_pos_mm is not None:
            self.GRIPPER_PLACE_POS_MM = place_pos_mm
            self.get_logger().info(f"Updated gripper place position to {place_pos_mm}mm")
    
    def get_gripper_config(self):
        """Returns current gripper configuration."""
        return {
            'open_pos_mm': self.GRIPPER_OPEN_POS_MM,
            'closed_pos_mm': self.GRIPPER_CLOSED_POS_MM,
            'place_pos_mm': self.GRIPPER_PLACE_POS_MM
        }

    def set_detection_config(self, min_frames_with_detections: int = None, 
                           max_detection_wait_time: float = None,
                           min_matching_frames_ratio: float = None,
                           min_matching_frames_absolute: int = None,
                           chip_matching_distance_threshold: float = None):
        """
        Update chip detection configuration for different detection conditions.
        
        Args:
            min_frames_with_detections: Minimum frames that must have detections
            max_detection_wait_time: Maximum time to wait for detections (seconds)
            min_matching_frames_ratio: Minimum ratio of frames that must match for robust detection
            min_matching_frames_absolute: Absolute minimum frames that must match
            chip_matching_distance_threshold: Distance threshold for matching chips across frames (meters)
        """
        if min_frames_with_detections is not None:
            self.MIN_FRAMES_WITH_DETECTIONS = min_frames_with_detections
            self.get_logger().info(f"Updated min frames with detections to {min_frames_with_detections}")
        if max_detection_wait_time is not None:
            self.MAX_DETECTION_WAIT_TIME = max_detection_wait_time
            self.get_logger().info(f"Updated max detection wait time to {max_detection_wait_time}s")
        if min_matching_frames_ratio is not None:
            self.MIN_MATCHING_FRAMES_RATIO = min_matching_frames_ratio
            self.get_logger().info(f"Updated min matching frames ratio to {min_matching_frames_ratio}")
        if min_matching_frames_absolute is not None:
            self.MIN_MATCHING_FRAMES_ABSOLUTE = min_matching_frames_absolute
            self.get_logger().info(f"Updated min matching frames absolute to {min_matching_frames_absolute}")
        if chip_matching_distance_threshold is not None:
            self.CHIP_MATCHING_DISTANCE_THRESHOLD = chip_matching_distance_threshold
            self.get_logger().info(f"Updated chip matching distance threshold to {chip_matching_distance_threshold}m")
    
    def get_detection_config(self):
        """Returns current detection configuration."""
        return {
            'min_frames_with_detections': self.MIN_FRAMES_WITH_DETECTIONS,
            'max_detection_wait_time': self.MAX_DETECTION_WAIT_TIME,
            'min_matching_frames_ratio': self.MIN_MATCHING_FRAMES_RATIO,
            'min_matching_frames_absolute': self.MIN_MATCHING_FRAMES_ABSOLUTE,
            'chip_matching_distance_threshold': self.CHIP_MATCHING_DISTANCE_THRESHOLD
        }

    def grasp_chip_fast_approach(self, robot_namespace: str, target_grasp_z_mm: float):
        """
        Fast approach to grasp a chip using a pre-measured Z height.
        This skips the surface detection and goes directly to the measured height.
        
        Args:
            robot_namespace: Robot namespace
            target_grasp_z_mm: Pre-measured Z height for grasping
        
        Returns:
            bool: True if grasp was successful, False otherwise
        """
        self.get_logger().info("--- Starting Fast Chip Grasp (Using Pre-measured Z Height) ---")
        
        # Get current pose
        if self.current_pose_robot1 is None:
            self.get_logger().error("No current pose available for grasp.")
            return False
            
        current_pose = self.current_pose_robot1
        approach_x = current_pose.position.x
        approach_y = current_pose.position.y
        approach_a = current_pose.orientation.x
        approach_b = current_pose.orientation.y
        approach_g = current_pose.orientation.z

        # 1. Preparations
        if not self.move_gripper(robot_namespace, position=self.GRIPPER_OPEN_POS_MM): return False
        self.set_blending(robot_namespace, 0)

        # 2. Move directly to the measured grasp height
        self.get_logger().info(f"Moving directly to measured grasp height: Z={target_grasp_z_mm:.2f}mm")
        if not self.move_pose_and_wait(robot_namespace, approach_x, approach_y, target_grasp_z_mm, 
                                     approach_a, approach_b, approach_g, 
                                     pos_tol_mm=0.1, orient_tol_deg=0.5, timeout_s=10.0):
            self.get_logger().error("Failed to move to measured grasp height.")
            return False
        time.sleep(0.2) # Settle

        # 3. Grasp Chip 
        self.get_logger().info(f"Closing gripper to {self.GRIPPER_CLOSED_POS_MM}mm")
        if not self.move_gripper(robot_namespace, position=self.GRIPPER_CLOSED_POS_MM):
             self.get_logger().error("Failed to close gripper.")
             return False
        time.sleep(1.0) # Allow gripper to physically close and stabilize grip

        # 4. Lift Chip
        self.get_logger().info("Lifting chip by 5.0mm")
        if not self.move_lin_rel_wrf(robot_namespace, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0):
             self.get_logger().error("Failed to lift chip.")
             return False
        time.sleep(0.5) # Settle

        self.get_logger().info("--- Fast Chip Grasp Sequence Presumed Successful ---")
        return True
    
    def stir_in_beaker(self, beaker_x: float, beaker_y: float, beaker_z: float, 
                      stir_duration_seconds: float, stir_frequency: float = 1.0,
                      circle_radius_mm: float = 8.0, circle_points: int = 8, 
                      move_z_distance_mm: float = 50.0, move_z_distance_time:float = 0.2,
                      beaker_name: str = "beaker", move_speed: int = 40):
        """
        Helper function to stir in a beaker at the specified position.
        
        Args:
            beaker_x, beaker_y, beaker_z: Position of the beaker
            stir_duration_seconds: How long to stir
            stir_frequency: How fast to stir
            circle_radius_mm: Radius of circular motion
            circle_points: Number of waypoints per circle
            move_z_distance_mm: Distance to move in and out before and after stirring
            beaker_name: Name for logging purposes
            move_speed: Speed for transport moves (restored after stirring)
        """
        # Move to beaker position
        self.set_joint_vel(self._robot1_ns, move_speed)
        self.move_pose_and_wait(self._robot1_ns, beaker_x, beaker_y, beaker_z + move_z_distance_mm, 
                                0, 90, 0)
        
        # Wait for pose feedback to settle
        rclpy.spin_once(self, timeout_sec=0.1)
        time.sleep(0.1)
        rclpy.spin_once(self, timeout_sec=0.1)

        # Perform circular motion inside the beaker
        self.get_logger().info(f"Starting circular motion inside the {beaker_name}.")
        # Print the pose before starting circle
        pose_before_circle = self.current_pose_robot1 
        self.get_logger().info(f"Current Pose: {pose_before_circle.position.x:.2f} / {pose_before_circle.position.y:.2f} / {pose_before_circle.position.z:.2f}.")

        # Perform the circular motion
        self.get_logger().info(f"Timed circular motion with radius {circle_radius_mm}mm for {stir_duration_seconds} seconds / {stir_frequency} Hz")
        success = self.move_circular_time_freq(circle_radius_mm, circle_points, 
                                               stir_duration_seconds, stir_frequency, 
                                               self._robot1_ns, 
                                               move_z_distance_mm=move_z_distance_mm, 
                                               move_z_time_seconds=move_z_distance_time)

        # Restore speed for transport
        self.set_joint_vel(self._robot1_ns, move_speed)

        if success:
            self.get_logger().info(f"{beaker_name.capitalize()} stirring completed successfully.")
        else:
            self.get_logger().warn(f"{beaker_name.capitalize()} stirring may have failed.")
            
        return success

    def clear_motion(self, robot_namespace: str, timeout_s: float = 10.0) -> bool:
        """Clear the robot's motion buffer."""       
        if not self.clear_motion_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().error(f"ClearMotion service for{robot_namespace} not available.")
            return False
        
        request = ClearMotion.Request()
        
        self.get_logger().info("Calling ClearMotion service to clear motion buffer...")
        future = self.clear_motion_client.call_async(request)
        
        rclpy.spin_until_future_complete(self, future, timeout_sec=timeout_s)
        
        result = future.result()
        if result is not None:
            if result.success:
                self.get_logger().info(f"ClearMotion service reported: {result.message}")            
                return True
            else:
                self.get_logger().error(f"ClearMotion service reported failure: {result.message}")    
                return False
        else:
            self.get_logger().error("No response from ClearMotion service (controller timed out).")
            return False

    def home(self, robot_namespace: str, timeout_s: float = 60.0) -> bool:
        """Calls the Home service to home the robot."""
        if not self.home_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().error(f"Home service for '{robot_namespace}' not available.")
            return False
        
        request = Home.Request()
        
        self.get_logger().info("Calling Home service to home robot...")
        future = self.home_client.call_async(request)
        
        rclpy.spin_until_future_complete(self, future, timeout_sec=timeout_s)
        
        result = future.result()
        if result is not None:
            if result.success:
                self.get_logger().info(f"Home service reported: {result.error_message}")
                return True
            else:
                self.get_logger().error(f"Home service reported failure: {result.error_message}")
                return False
        else:
            self.get_logger().error("No response from Home service (controller timed out).")
            return False

            
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

