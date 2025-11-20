#!/usr/bin/env python3

import math
import time
import serial

import numpy as np
from scipy.spatial.transform import Rotation as R


import rclpy
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from geometry_msgs.msg import Pose
from std_srvs.srv import Trigger
from custom_interfaces.msg import PoseStampedArray

from meca_controller.meca_control import MecaControl
from meca_controller.meca_settings import ROBOT1

# CONFIGURATION

# --- Gripper Configuration ---
GRIPPER_OPEN_POS_MM   = 18.0    # Gripper position when fully open
GRIPPER_CLOSED_POS_MM = 9.5     # Gripper position when closed on chip FIXME: would be great to get this out of the image.
GRIPPER_CLOSED_POS_TIGHT_MM = 8.5
GRIPPER_PLACE_POS_MM  = 14.0    # Gripper position when placing chip (partially open)


class MecaDemoDevelop(MecaControl):
    def __init__(self):
        settings = {
            "move_speed": ROBOT1['move_speed'],
            "port_usb_relay": "/dev/ttyUSB0",
            "gripper": {
                "open_pos_mm": GRIPPER_OPEN_POS_MM,
                "closed_pos_mm": GRIPPER_CLOSED_POS_MM,
                "closed_pos_tight_mm": GRIPPER_CLOSED_POS_TIGHT_MM,
                "place_pos_mm": GRIPPER_PLACE_POS_MM
            },
            "positions": {
                "home": [0, 0, 0, 0, 0, 0],
                "chip_view": [180.0, 110.0, 136.577, 0.0, 90.0, 0.0],
                "stir_beaker_1": [193.0, 18.5, 121.8, 0.0, 90.0, 0.0],
                "stir_beaker_2": [193.0, -78.0, 121.8, 0.0, 90.0, 0.0],
                "N2_gun": [25.8, -204.5, 83.5, 75.6, -13.5, -96.0]
            },
            "process": {
                "n2_gun":{
                    "time_on_s": 10.0,
                },
                "beaker":{
                    "stir_radius_mm": 8.0,
                    "stir_circle_points": 16,
                    "stir_frequency": 1,
                    "move_z_distance_mm": 60.0, # Distance to move up
                    "beaker_height_mm": 121.8,  # Height of beakers
                    "stir_duration_beaker_1_sec": 40.0,
                    "stir_duration_beaker_2_sec": 10.0,
                },
            }
        }
        super().__init__('meca_develop', ROBOT1['namespace'], settings=settings, enable_moveit=True)

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
         # List to store all discovered chip poses
        self.latest_chip_poses = []
        self.get_logger().info("Subscribed to /chip_pose_calculator/detected_chip_world_poses with sensor QoS")
        
        self._initialize_clients()
        
        # Store measured Z height for reuse
        self.measured_grasp_z_height = None
        
    def _initialize_clients(self):
        """ Initialize clients and preload some clients that we want to have access to fast.
        """
        # Add refresh detection service client
        self.refresh_detection_client = self.create_client(
            Trigger, 
            '/chip_pose_calculator/refresh_detection'
        )
        self.get_logger().info("Created refresh detection service client")

        _ = self._client_move_pose # Warmup cache.

        
    def run(self):
        """ Develop chips utilizing two beakers, and a nitrogen gun for the drying step.
        """
        # Safety config
        self.set_joint_vel(self._settings['move_speed'])
        self.set_gripper_force(5)
        self.move_gripper(position=self._settings['gripper']['open_pos_mm'])

        self.move_joints(self._settings['positions']['home'])
        self.home()

        camera_pose = self._settings['positions']['chip_view']
        pose_reached = self.move_pose_and_wait(*camera_pose)
        
        if pose_reached:
            self.get_logger().info("Pose reached successfully.")
        else:
            self.get_logger().error("Failed to reach pose.")

        self.spin_until_timer(3.0) # wait for the chip detection to become active
        
        # Trigger refresh after robot moves to new position
        self.get_logger().info("Triggering chip detection refresh after robot movement...")
        refresh_success = self.refresh_chip_detection(wait_for_completion=True, timeout_seconds=10)
        if refresh_success:
            self.get_logger().info("Chip detection refresh completed successfully")
        else:
            self.get_logger().warn("Chip detection refresh may have failed, continuing anyway")
        
        input("Press Enter to continue...")
        
        # Check if the topic is publishing
        self.get_logger().info("Check available topics...")
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
        
        self.get_logger().info("Waiting for stable chip poses from detection node...")
        
        # Wait for stable chip poses from the detection node
        max_wait_time = 10.0  # Maximum time to wait for detections
        start_time = time.time()
        
        while rclpy.ok() and (time.time() - start_time) < max_wait_time:
            #FIXME: check if spin_until_timer could work here
            rclpy.spin_once(self, timeout_sec=0.1)
            time.sleep(0.1)
            
            if len(self.latest_chip_poses) > 0:
                self.get_logger().info(f"Received {len(self.latest_chip_poses)} stable chip poses from detection node")
                break
            else:
                self.get_logger().info("Waiting for stable chip poses...")

        if not rclpy.ok():
            self.get_logger().error("Node shutting down.")
            return
            
        if len(self.latest_chip_poses) == 0:
            self.get_logger().error(f"Timeout waiting for chip detections. No stable poses received.")
            return

        robust_chip_poses = self.latest_chip_poses
        
        self.get_logger().info(f"Found {len(robust_chip_poses)} stable chip(s) to process.")
        for i, pose in enumerate(robust_chip_poses):
            self.get_logger().info(f"  Chip {i+1}: P({pose.pose.position.x*1000:.1f}, {pose.pose.position.y*1000:.1f}, {pose.pose.position.z*1000:.1f})mm")

        # Process each detected chip
        for i, chip_pose_msg in enumerate(robust_chip_poses):
            self.get_logger().info(f"Processing chip {i+1}/{len(robust_chip_poses)}")           

            # Configuration for stirring
            circle_radius_mm = self._settings['process']['beaker']['stir_radius_mm']  # 8mm radius
            circle_points = self._settings['process']['beaker']['stir_circle_points']      # 16 points 
            move_z_distance_mm = self._settings['process']['beaker']['move_z_distance_mm']  # Distance to move up (180.0 - 121.8 = 58.2mm)
            beaker_height_mm = self._settings['process']['beaker']['beaker_height_mm']  # Height of beakers

            # Safe height for moving between beakers
            safe_height_mm = self._settings['process']['beaker']['beaker_height_mm'] + self._settings['process']['beaker']['move_z_distance_mm']  
            self.get_logger().info(f"Chip pose: {chip_pose_msg}")

            # Transform chip pose to TCP target for grasping
            grasp_target = self.transform_chip_pose_to_tcp_target(chip_pose_msg, target_z_mm=95.0)
            if grasp_target is None:
                self.get_logger().error(f"Failed to transform chip pose for grasping.")
                self.goto_chip_view()
                continue
            self.get_logger().info(f"Grasp target: {grasp_target}")
            # Call the goto_chip_pose function with the pre-transformed target
            success = self.goto_chip_pose(tcp_target=grasp_target)
            if success:
                self.get_logger().info("Successfully moved to chip pose.")
                # Now grasp the chip from the current position
                if i == 0:
                    # First chip: measure Z height with full surface detection
                    grasp_success = self.grasp_chip_fixed_location_by_lifting()
                    if grasp_success:
                        self.get_logger().info(f"Measured grasp Z height: {self.measured_grasp_z_height:.2f}mm")
                    else:
                        self.get_logger().error("Failed to grasp the first chip.")
                        self.goto_chip_view()
                        continue  # Skip to next chip if grasp failed
                else:
                    # Subsequent chips: use measured Z height for fast approach
                    if self.measured_grasp_z_height is not None:
                        grasp_success = self.grasp_chip_fast_approach(self.measured_grasp_z_height)
                    else:
                        self.get_logger().error("No measured Z height available for subsequent chip.")
                        self.goto_chip_view()
                        continue
                
                self.set_joint_vel(self._settings['move_speed'])
                if grasp_success:
                    self.get_logger().info("Successfully grasped the chip.")
                else:
                    self.get_logger().error("Failed to grasp the chip.")
                    self.goto_chip_view()
                    continue  # Skip to next chip if grasp failed
            else:
                self.get_logger().error("Failed to move to chip pose.")
                self.goto_chip_view()
                continue  # Skip to next chip if movement failed

            # Only proceed if grasp_success is True
            self.set_joint_vel(self._settings['move_speed'])
            self.move_joints(np.array(self._settings['positions']['home']))
            self.move_gripper(position=self._settings['gripper']['closed_pos_tight_mm'])

            # --- Stir in first beaker ---
            first_beaker_success = self.stir_in_beaker(
                beaker_x=self._settings['positions']['stir_beaker_1'][0], 
                beaker_y=self._settings['positions']['stir_beaker_1'][1], 
                beaker_z=self._settings['positions']['stir_beaker_1'][2],
                stir_duration_seconds=self._settings['process']['beaker']['stir_duration_beaker_1_sec'], 
                stir_frequency=self._settings['process']['beaker']['stir_frequency'], 
                circle_radius_mm=self._settings['process']['beaker']['stir_radius_mm'],
                circle_points=self._settings['process']['beaker']['stir_circle_points'],
                move_z_distance_mm=self._settings['process']['beaker']['move_z_distance_mm'], 
                move_z_distance_time = 0.3,
                beaker_name="beaker_1",
                move_speed=self._settings['move_speed'],
                block=False
            )

            if not first_beaker_success:
                self.get_logger().error("Failed to stir in first beaker, return to chip view.")
                self.goto_chip_view()
                continue
            
            # --- Go to second beaker ---
            self.move_pose(self._settings['positions']['stir_beaker_1'][0],  
                           self._settings['positions']['stir_beaker_1'][1], 
                           safe_height_mm, 0, 90, 0)
            self.move_pose(self._settings['positions']['stir_beaker_2'][0], 
                           self._settings['positions']['stir_beaker_2'][1], 
                           safe_height_mm, 0, 90, 0)

            # --- Stir in second beaker ---
            second_beaker_success = self.stir_in_beaker(
                beaker_x=self._settings['positions']['stir_beaker_2'][0], 
                beaker_y=self._settings['positions']['stir_beaker_2'][1], 
                beaker_z=beaker_height_mm,
                stir_duration_seconds=self._settings['process']['beaker']['stir_duration_beaker_2_sec'],
                stir_frequency=self._settings['process']['beaker']['stir_frequency'], 
                circle_radius_mm=self._settings['process']['beaker']['stir_radius_mm'],
                circle_points=self._settings['process']['beaker']['stir_circle_points'],
                move_z_distance_mm=self._settings['process']['beaker']['move_z_distance_mm'], 
                move_z_distance_time = 0.3,
                beaker_name="beaker_2",
                move_speed=self._settings['move_speed'],
                block=True
            )
            self.get_logger().info("Stirring in second beaker complete.")
            self.set_joint_vel(self._settings['move_speed'])
            if not second_beaker_success:
                self.get_logger().error("Failed to stir in second beaker.")
                self.move_pose_and_wait(self._settings['positions']['stir_beaker_2'][0], 
                                        self._settings['positions']['stir_beaker_2'][1], 
                                        safe_height_mm, 0, 90, 0)
            
            # go to N2 gun
            pose_N2 = self._settings['positions']['N2_gun']
            pose_reached = self.move_pose_and_wait(pose_N2[0], pose_N2[1], pose_N2[2], pose_N2[3], pose_N2[4], pose_N2[5])
            if pose_reached:
                self.get_logger().info("N2 pose reached successfully.")
            else:
                self.get_logger().error("Failed to reach N2 pose.")

            # Switch USB relay before going to N2 gun
            self.switch_usb_relay(delay_on_s=self._settings['process']['n2_gun']['time_on_s'])
            self.move_joints(np.array(self._settings['positions']['home']))
            self.home()

            # Use the same grasp_target for placement (same location, different height)
            # The place_chip_at_fixed_location function will handle the Z offset internally
            placement_success = self.place_chip_at_fixed_location(
                chip_drop_pose_params=grasp_target)
            if placement_success:
                self.get_logger().info("Chip placement sequence successful - chip placed back at original location.")
            else:
                self.get_logger().error("Chip placement sequence failed.")
                
            self.set_joint_vel(self._settings['move_speed'])
            
        self.set_joint_vel(self._settings['move_speed'])

        #input("Press Enter here to move back to chip view...")
        pose_reached = self.move_pose_and_wait(*camera_pose)
        self.get_logger().info("Finished processing chips.")

    def goto_chip_view(self):
        pose = self._settings['positions']['chip_view']
        
        self.get_logger().info(f"Goto chip view pose: {pose}")
        pose_reached = self.move_pose_and_wait(*pose)
        
        if pose_reached:
            self.get_logger().info("ChipView pose reached successfully.")
        else:
            self.get_logger().error("Failed to reach pose.") 

    def grasp_chip_fixed_location_by_lifting(
        self,
        initial_press_depth_mm: float = 5.0, # Press down this much past approach to ensure contact
        lift_step_mm: float = 0.1,           # Small steps for lifting to find release
        max_lift_search_mm: float = 3.0,
        # We find contact with the surface by observing the torque of the arm.
        # We found Joint 5 (index 4) to be the best primary indicator.
        #   When pressed, J5 torque is high (e.g., > 25%). 
        #   When free, it's low (e.g., < 10% and often negative).
        contact_joint_indices: list[int] = [4],
        torque_delta_release_threshold_percent: float = 0.8,   # Small change threshold - indicates we're in free-air
        press_torque_increase_threshold_percent: float = 10.0, # J5 must increase by at least 10% from free-air
        significant_drop_threshold_percent: float = 15.0,      # Large drop threshold - indicates we've started releasing
        grasp_height_above_surface_mm: float = 0.2,            # e.g. for a 0.5mm thick chip
        gripper_open_pos_mm: float = None,
        gripper_closed_pos_mm: float = None,
        lift_chip_height_mm: float = 5.0
    ) -> bool:
        # Use centralized gripper configuration if not specified
        if gripper_open_pos_mm is None:
            gripper_open_pos_mm = self._settings['gripper']['open_pos_mm']
        if gripper_closed_pos_mm is None:
            gripper_closed_pos_mm = self._settings['gripper']['closed_pos_mm']
            
        self.get_logger().info("--- Starting Chip Grasp (Press Down, Lift to Detect Surface) ---")
        
        if self.current_pose_robot is None:
            self.get_logger().error("No current pose available for grasp.")
            return False
            
        current_pose = self.current_pose_robot
        approach_x = current_pose.position.x
        approach_y = current_pose.position.y
        approach_z_start = current_pose.position.z
        approach_a = current_pose.orientation.x
        approach_b = current_pose.orientation.y
        approach_g = current_pose.orientation.z

        # 1. Preparations
        if not self.move_gripper(position=gripper_open_pos_mm): return False
        self.set_blending(0)

        # 2. We're already at the approach pose (from goto_chip_pose), so just settle
        self.get_logger().info(f"Already at approach pose Z: {approach_z_start:.2f}")
        self.spin_until_timer(0.3) # Settle

        rt_data_baseline = self.get_rt_joint_torq(False, True, 1.0)    
        if not rt_data_baseline or not rt_data_baseline.success or len(rt_data_baseline.torques) == 0:
            self.get_logger().error("Failed to get baseline free-air torques."); return False

        baseline_torques_free_air = np.array(rt_data_baseline.torques)
        self.get_logger().info(f"Baseline free-air torques (%): {baseline_torques_free_air.round(2)}")

        # 3. Press Down to Ensure Contact
        self.set_joint_vel(2)
        self.get_logger().info(f"Pressing down by {initial_press_depth_mm:.2f}mm to ensure contact.")
        res = self.move_lin_rel_wrf(0.0, 0.0, -initial_press_depth_mm, 0.0, 0.0, 0.0)
        self.wait_idle()
        if not res:
             self.get_logger().error("Failed to execute press_down move."); return False
        self.spin_until_timer(0.5) # Settle 

        rt_data_pressing = self.get_rt_joint_torq(False, True, 1.0)
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
            self.move_lin_rel_wrf(0.0, 0.0, initial_press_depth_mm, 0.0, 0.0, 0.0)
            self.wait_idle()
            return False

        # 4. Lift Step-by-Step to Detect Surface Release
        self.get_logger().info(f"Lifting by {lift_step_mm}mm steps to detect surface release (large drop > {significant_drop_threshold_percent}% then small changes < {torque_delta_release_threshold_percent}%)...")
        
        surface_z_at_release = None
        previous_step_torques = np.copy(torques_at_full_press) # Torques when fully pressed
        pose_at_surface_contact = self.current_pose_robot # Pose when fully pressed (before starting to lift)
        if not pose_at_surface_contact: # Should have pose here
            self.get_logger().error("Critical: Lost pose feedback before starting lift detection."); return False

        max_lift_steps = int(max_lift_search_mm / lift_step_mm) + 1
        has_had_significant_drop = False  # Track if we've had a large drop from pressed state
        
        for step_num in range(max_lift_steps):
            # Store pose *before* this specific lift step - this will be the surface Z if release is detected *after* this lift
            pose_before_this_lift_segment = self.current_pose_robot
            if not pose_before_this_lift_segment:
                self.get_logger().warn(f"No pose feedback before lift step {step_num + 1}. Using last known Z.")
                # Fallback to using Z from pose_at_surface_contact + accumulated lift
                # This can introduce error if a previous pose was missed.
                temp_z = pose_at_surface_contact.position.z + (step_num * lift_step_mm)
                pose_before_this_lift_segment = Pose()
                pose_before_this_lift_segment.position.z = temp_z

            res = self.move_lin_rel_wrf(0.0, 0.0, +lift_step_mm, 0.0, 0.0, 0.0)
            self.wait_idle()
            if not res:
                self.get_logger().error("MoveLinRelWrf lift step failed. Aborting."); 
                return False

            self.spin_until_timer(0.15) # Settle and allow feedback update

            rt_data_current = self.get_rt_joint_torq(False, True, 0.5)
            if not rt_data_current or not rt_data_current.success or len(rt_data_current.torques) == 0:
                self.get_logger().warn("Failed to get current torques during lift search."); 
                continue
            
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
                if self.current_pose_robot:
                     surface_z_at_release = self.current_pose_robot.position.z - lift_step_mm
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
        self.measured_grasp_z_height = target_grasp_z
        
        if not self.move_pose_and_wait(approach_x, approach_y, target_grasp_z, 
                                       approach_a, approach_b, approach_g, 
                                       pos_tol_mm=0.1, orient_tol_deg=0.5, timeout_s=10.0): # Tighter tolerance
            self.get_logger().error("Failed to move to final grasp height."); return False
        
        self.spin_until_timer(0.2) # Settle

        # 6. Grasp Chip 
        self.get_logger().info(f"Closing gripper to {gripper_closed_pos_mm}mm")
        if not self.move_gripper(position=gripper_closed_pos_mm): # Ensure this blocks or add appropriate wait
             self.get_logger().error("Failed to close gripper.")
             return False
        self.spin_until_timer(1.0) # Allow gripper to physically close and stabilize grip

        # 7. Lift Chip
        self.get_logger().info(f"Lifting chip by {lift_chip_height_mm}mm")
        res = self.move_lin_rel_wrf(0.0, 0.0, lift_chip_height_mm, 0.0, 0.0, 0.0)
        self.wait_idle()
        if not res:
             self.get_logger().error("Failed to lift chip.")
             return False
        self.spin_until_timer(0.5) # Settle

        self.get_logger().info("--- Chip Grasp by Lifting Sequence finished ---")
        return True

    def touchdown_torque_plotter(self,
                                 approach_pose_params: tuple = (190.0, 158.943, 96.577, 0.0, 90.0, 0.0),
                                 step_mm: float = 0.2, 
                                 N: int = 10):
        """Plots torque values for specified joints vs Z.
        
        Used to find the best joint to use for contact detection and calibrate the torque threshold.
        """
        import matplotlib.pyplot as plt
        torques_all = []
        dz_values = []
        self.get_logger().info(f"Moving to starting pose...")
        self.move_pose_and_wait(approach_pose_params[0], approach_pose_params[1], approach_pose_params[2], 
                                approach_pose_params[3], approach_pose_params[4], approach_pose_params[5])
        
        self.spin_until_timer(0.5)
        
        rt_data = self.get_rt_joint_torq(False, True, 0.5)
        torques_all.append(np.array(rt_data.torques))
        dz_values.append(0.0)  # Initial position
        
        # Move down
        self.get_logger().info(f"Moving down...")
        for i in range(1, N):
            dz = -i * step_mm
            self.move_lin_rel_wrf(0.0, 0.0, -step_mm, 0.0, 0.0, 0.0)
            self.wait_idle()
            self.spin_until_timer(0.1)
            rt_data = self.get_rt_joint_torq(False, True, 0.5)
            self.get_logger().info(f"Torques: {np.array(rt_data.torques)}")
            torques_all.append(np.array(rt_data.torques))
            dz_values.append(dz)
            
        # Move back up
        self.get_logger().info(f"Moving back up...")
        self.set_joint_vel(2) # Slow and careful for probing
        for i in range(1, N):
            dz = i * step_mm
            self.move_lin_rel_wrf(0.0, 0.0, step_mm, 0.0, 0.0, 0.0)
            self.wait_idle()
            self.spin_until_timer(1.0)
            rt_data = self.get_rt_joint_torq(False, True, 0.5)
            self.get_logger().info(f"Torques: {np.array(rt_data.torques)}")
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
                                     chip_drop_pose_params: tuple,
                                     approach_offset_z_mm: float = 3.5,
                                     gripper_open_pos_mm: float = None,
                                     retract_after_drop_mm: float = 10.0,
                                     placement_joint_vel_percent: float = 5.0
                                     ) -> bool:
        """
        Places a held chip at a specified location by approaching, moving to a drop height,
        opening the gripper, and retracting.

        Args:
            chip_drop_pose_params: Tuple (x,y,z,a,b,g) where the TCP should be
                                   (e.g., 1mm above surface) when opening the gripper.
            approach_offset_z_mm: Initial approach height above the final drop_z.
            gripper_open_pos_mm: The gripper position when the gripper opens (Centralized config if not specified)
            retract_after_drop_mm: How much to retract after the chip is dropped.
            placement_joint_vel_percent: Velocity percentage for the movement joints.
        Returns:
            bool: True if chip placement sequence presumed successful, False otherwise.
        """

        if gripper_open_pos_mm is None:
            gripper_open_pos_mm = self._settings['gripper']['place_pos_mm']
            
        self.get_logger().info("--- Starting Chip Placement Sequence ---")
        drop_x, drop_y, drop_z, drop_a, drop_b, drop_g = chip_drop_pose_params

        # --- 1. Preparation ---
        # Assume chip is already held (gripper closed appropriately)
        self.set_blending(0) # For precise positioning

        # --- 2. Approach Above Destination ---
        self.get_logger().info(f"Moving to approach pose: Z={drop_z:.2f}")
        if not self.move_pose_and_wait(drop_x, drop_y, drop_z, 
                                       drop_a, drop_b, drop_g, 
                                       timeout_s=15.0, pos_tol_mm=1.0, orient_tol_deg=1.0):
            self.get_logger().error("Failed to reach approach pose above destination.")
            return False
        self.spin_until_timer(0.2)# Settle

        # --- 3. Move to Final Drop Height ---
        self.set_joint_vel(placement_joint_vel_percent) 
        self.get_logger().debug(f"Moving to final drop height")
        res = self.move_lin_rel_wrf(0.0, 0.0, -approach_offset_z_mm, 0.0, 0.0, 0.0)
        self.wait_idle()
        if not res:
            self.get_logger().error("Failed to move to final drop height.")
            return False
        self.spin_until_timer(0.5); # Settle at drop height before opening gripper

        # --- 4. Open Gripper (Release Chip) ---
        self.get_logger().debug(f"Opening gripper to {gripper_open_pos_mm}mm to release chip.")
        if not self.move_gripper(position=gripper_open_pos_mm):
             self.get_logger().error("Failed to open gripper.")
             return False
        # Allow chip to drop
        self.spin_until_timer(0.5)

        # --- 5. Retract Upwards ---
        self.get_logger().debug(f"Retracting upwards by {retract_after_drop_mm}mm.")
        res = self.move_lin_rel_wrf(0.0, 0.0, retract_after_drop_mm, 0.0, 0.0, 0.0)
        self.wait_idle()

        if not res:
             self.get_logger().error("Failed to retract after drop.")
             # Since the chip was successfully dropped, we consider this a success
             # but log the retraction failure for monitoring
             self.get_logger().warn("Chip was placed but robot failed to retract properly.")
             return True
        else:
            self.spin_until_timer(0.2) # Settle after retract

        self.get_logger().debug("--- Chip Placement Sequence Finished ---")
        return True
    
    def chip_poses_callback(self, msg):
        """ Store the stable poses from the detection node.
        """
        self.latest_chip_poses = msg.poses
    
    def refresh_chip_detection(self, wait_for_completion=True, timeout_seconds=10):
        try:
            # Wait for service to be available
            if not self.refresh_detection_client.wait_for_service(timeout_sec=5.0):
                self.get_logger().error("Refresh detection service not available")            
                return False
            
            # Clear existing poses before refresh
            self.latest_chip_poses = []
            
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
                        # Actively wait for detection to complete with early exit
                        self.get_logger().info(f"Waiting up to {timeout_seconds} seconds for detection to complete...")
                        start_time = time.time()
                        
                        while (time.time() - start_time) < timeout_seconds:
                            # Spin once to process callbacks
                            rclpy.spin_once(self, timeout_sec=0.1)
                            
                            # Check if we received new poses
                            if self.latest_chip_poses:
                                elapsed_time = time.time() - start_time
                                self.get_logger().info(f"Fresh detection complete in {elapsed_time:.2f}s! Got {len(self.latest_chip_poses)} chip poses")
                                return True
                            
                            # Small sleep to avoid busy waiting
                            time.sleep(0.1)
                        
                        # Timeout reached
                        self.get_logger().warn(f"No chip poses received after {timeout_seconds}s timeout")
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
        self.get_logger().info("Attempting to lookup TCP offset from TF...")
        
        # Try multiple times with increasing delays
        max_retries = 5
        tcp_offset_local = None
        
        for attempt in range(max_retries):
            try:
                transform = self.tf_buffer.lookup_transform('meca_axis_6_link', 'tweezer_tcp', rclpy.time.Time())
                tcp_offset_local = np.array([
                    transform.transform.translation.x,
                    transform.transform.translation.y,
                    transform.transform.translation.z
                ])
                self.get_logger().info(f"Retrieved TCP offset from TF: {tcp_offset_local} meters")
                break  # Success, exit retry loop
                
            except Exception as e:
                if attempt < max_retries - 1:
                    self.get_logger().warn(f"TCP transform lookup failed (attempt {attempt + 1}/{max_retries}): {e}")
                    self.get_logger().warn("Waiting before retry...")
                    self.spin_until_timer(0.5 * (attempt + 1))  # Increasing delay
                else:
                    self.get_logger().error(f"TCP transform lookup failed after {max_retries} attempts: {e}")
                    self.get_logger().error("Cannot proceed without TCP transform - check if tcp_tf.launch.py is running")
                    return None
        
        if tcp_offset_local is None:
            self.get_logger().error("Failed to get TCP offset - cannot calculate robot pose")
            return None
            
        self.get_logger().info(f"Using TCP offset: {tcp_offset_local} meters")

        self.get_logger().info("--- Calculating Position ---")
        
        # To get the correct position, we use a standard position that 
        # points the tool straight down but aligned with the chip
        calculation_orientation = R.from_euler('xyz', [180, 0, chip_yaw_deg+90.0], degrees=True)
        self.get_logger().info(f"Using orientation for calculation: (180, 0, {chip_yaw_deg:.2f})")

        # Define the target position for the TCP
        P_tcp_target_m = np.array([
            chip_pose_msg.pose.position.x,
            chip_pose_msg.pose.position.y,
            target_z_mm / 1000.0  # Convert target Z from mm to meters
        ])

        # Calculate the flange position
        tcp_offset_world = calculation_orientation.apply(tcp_offset_local)
        P_flange_target_m = P_tcp_target_m - tcp_offset_world
        
        self.get_logger().info(f"Calculated Flange Position (X, Y, Z meters): {P_flange_target_m}")

        # Define hardcoded orientation angles
        manual_alpha = 90.0
        manual_beta = 180.0+chip_yaw_deg
        manual_gamma = -90.0

        self.get_logger().info(f"Override with manual orientation: ({manual_alpha:.2f}, {manual_beta:.2f}, {manual_gamma:.2f})")

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
        usb_relay = serial.Serial(self._settings['port_usb_relay'], 9600)
        if usb_relay.is_open:
            self.get_logger().info(str(usb_relay))
            on_cmd = b'\xA0\x01\x01\xa2'
            off_cmd =  b'\xA0\x01\x00\xa1'


            usb_relay.write(on_cmd )
            self.spin_until_timer(delay_on_s)
            usb_relay.write(off_cmd)
            usb_relay.close()

    def goto_chip_pose(self, chip_pose_msg=None, target_z_mm=95.0, tcp_target=None):
        """
        Move the robot to a chip pose. Uses provided tcp_target or transform chip_pose_msg
        
        Args:
            chip_pose_msg: The chip pose message containing position and orientation (optional if tcp_target provided)
            target_z_mm: Target Z height in mm (default 95.0, used only if tcp_target not provided)
            tcp_target: Pre-transformed TCP target pose tuple (x, y, z, alpha, beta, gamma) (optional if chip_pose_msg provided)
        
        Returns:
            bool: True if movement was successful, False otherwise
        """

        if chip_pose_msg is None and tcp_target is None:
            self.get_logger().error("Chip Pose: Either chip_pose_msg or tcp_target must be provided.")
            return False
        
        if tcp_target is not None:
            x, y, z, alpha, beta, gamma = tcp_target
        else: # Transform chip pose to TCP target
            tcp_target = self.transform_chip_pose_to_tcp_target(chip_pose_msg, target_z_mm)
            if tcp_target is None:
                self.get_logger().error("Chip Pose: Failed to transform chip pose to TCP target.")
                return False
            x, y, z, alpha, beta, gamma = tcp_target
        
        if self.move_pose_and_wait(x, y, z, alpha, beta, gamma):
            self.get_logger().info("Chip Pose: Robot reached target pose.")
            return True
        else:
            self.get_logger().error("Chip Pose: Robot could not reach the target pose.")
            return False
    
    def set_gripper_config(self, open_pos_mm: float = None, closed_pos_mm: float = None, place_pos_mm: float = None):
        """
        Update gripper configuration for different chip sizes.
        
        Args:
            open_pos_mm: Gripper position when fully open (default: keep current)
            closed_pos_mm: Gripper position when closed on chip (default: keep current)
            place_pos_mm: Gripper position when placing chip (default: keep current)
        """
        if open_pos_mm is not None:
            self._settings['gripper']['open_pos_mm'] = open_pos_mm
            self.get_logger().info(f"Updated gripper open position to {open_pos_mm}mm")
        if closed_pos_mm is not None:
            self._settings['gripper']['closed_pos_mm'] = closed_pos_mm
            self.get_logger().info(f"Updated gripper closed position to {closed_pos_mm}mm")
        if place_pos_mm is not None:
            self._settings['gripper']['place_pos_mm'] = place_pos_mm
            self.get_logger().info(f"Updated gripper place position to {place_pos_mm}mm")
    
    def get_gripper_config(self):
        """Returns current gripper configuration."""
        return {
            'open_pos_mm': self._settings['gripper']['open_pos_mm'],
            'closed_pos_mm': self._settings['gripper']['closed_pos_mm'],
            'place_pos_mm': self._settings['gripper']['place_pos_mm']
        }


    def grasp_chip_fast_approach(self, target_grasp_z_mm: float):
        """
        Fast approach to grasp a chip using a pre-measured Z height.
        This skips the surface detection and goes directly to the measured height.
        
        Args:
            target_grasp_z_mm: Pre-measured Z height for grasping
        
        Returns:
            bool: True if grasp was successful, False otherwise
        """
        self.get_logger().info("--- Starting Fast Chip Grasp (Using Pre-measured Z Height) ---")
        
        # Get current pose
        if self.current_pose_robot is None:
            self.get_logger().error("No current pose available for grasp.")
            return False
            
        current_pose = self.current_pose_robot
        approach_x = current_pose.position.x
        approach_y = current_pose.position.y
        approach_a = current_pose.orientation.x
        approach_b = current_pose.orientation.y
        approach_g = current_pose.orientation.z

        if not self.move_gripper(position=self._settings['gripper']['open_pos_mm']):
            self.get_logger().error("Failed to open gripper.")
            return False
        self.set_blending(0)

        # Move to the measured grasp height
        self.get_logger().info(f"Moving directly to measured grasp height: Z={target_grasp_z_mm:.2f}mm")
        if not self.move_pose_and_wait(approach_x, approach_y, target_grasp_z_mm, 
                                       approach_a, approach_b, approach_g, 
                                       pos_tol_mm=0.1, orient_tol_deg=0.5, timeout_s=10.0):
            self.get_logger().error("Failed to move to measured grasp height.")
            return False
        self.spin_until_timer(0.2) # Settle

        # Grasp Chip 
        self.get_logger().info(f"Closing gripper to {self._settings['gripper']['closed_pos_mm']}mm")
        if not self.move_gripper(position=self._settings['gripper']['closed_pos_mm']):
             self.get_logger().error("Failed to close gripper.")
             return False
        self.spin_until_timer(1.0) # Settle grasp

        # Lift Chip
        self.get_logger().info("Lifting chip by 5.0mm")
        res = self.move_lin_rel_wrf(0.0, 0.0, 5.0, 0.0, 0.0, 0.0)

        self.wait_idle()
        if not res:
             self.get_logger().error("Failed to lift chip.")
             return False
        self.spin_until_timer(0.5) # Settle

        self.get_logger().info("--- Fast Chip Grasp Sequence Presumed Successful ---")
        return True
    
    def stir_in_beaker(self, beaker_x: float, beaker_y: float, beaker_z: float, 
                       stir_duration_seconds: float, stir_frequency: float = 1.0,
                       circle_radius_mm: float = 8.0, circle_points: int = 8, 
                       move_z_distance_mm: float = 50.0, move_z_distance_time:float = 0.2,
                       beaker_name: str = "beaker", move_speed: int = 40, block: bool = True):
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
            block: Whether to block until the stirring is complete. Default True.
        """
        # Move to beaker position
        self.set_joint_vel(move_speed)

        self.move_pose(beaker_x, beaker_y, beaker_z + move_z_distance_mm, 
                       0, 90, 0)

        self.get_logger().info(f"Starting circular motion inside the {beaker_name}.")

        # Perform the circular motion
        self.get_logger().info(f"Timed circular motion with radius {circle_radius_mm}mm for {stir_duration_seconds} seconds / {stir_frequency} Hz")
        
        start_pose = self.create_pose(beaker_x, beaker_y, beaker_z + move_z_distance_mm, 0, 90, 0)

        success = self.move_circular_time_freq(radius_mm=circle_radius_mm, num_points=circle_points, 
                                               duration_seconds=stir_duration_seconds, 
                                               start_pose=start_pose,
                                               frequency=stir_frequency,
                                               clockwise=True,
                                               move_z_distance_mm=move_z_distance_mm, 
                                               move_z_time_seconds=move_z_distance_time,
                                               block=block)

        # Restore speed for transport
        self.set_joint_vel(move_speed)

        if success:
            self.get_logger().info(f"{beaker_name.capitalize()} stirring completed successfully.")
        else:
            self.get_logger().warn(f"{beaker_name.capitalize()} stirring may have failed.")
            
        return success


def main(args=None):
    """
    Run this after you have started up the meca_driver node and are connected to the robot (and have started the motion planner):
    ros2 run meca_controller meca_control
    """

    rclpy.init(args=args)
    node = MecaDemoDevelop()
    
    try:
        node.wait_for_initialization()
        node.run()
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.destroy_node()
        rclpy.try_shutdown()
