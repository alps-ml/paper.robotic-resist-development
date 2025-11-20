import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge # Package to convert ROS images to OpenCV
import cv2
import numpy as np 
from geometry_msgs.msg import PointStamped, PoseStamped, Quaternion
from std_msgs.msg import String
from std_srvs.srv import Trigger

from rclpy.time import Duration
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
import tf2_ros
import math
from scipy.spatial.transform import Rotation as R
import tf2_geometry_msgs.tf2_geometry_msgs
from custom_interfaces.msg import PoseStampedArray
import time



class chip_detection(Node):
    # Multi-scale detection parameters (internal algorithm settings)
    DEFAULT_BLOCK_SIZES = [5, 7, 9]
    DEFAULT_KERNEL_SIZES = [3, 5]
    
    # Preprocessing parameters (internal algorithm settings)
    BILATERAL_D = 9
    BILATERAL_SIGMA = 75
    THRESH_C = 2
    
    # Contour filtering parameters (internal algorithm settings)
    LARGE_CONTOUR_THRESHOLD = 4000
    
    # Multi-scale detection parameters (internal algorithm settings)
    MERGE_DISTANCE_THRESHOLD = 20
    CONFIDENCE_THRESHOLD = 1
    
    def __init__(self, image_topic="/camera/camera/color/image_rect_raw"):
        super().__init__('chip_pose_calculator')
        
        # Declare all parameters
        self.declare_parameters(
            namespace='',
            parameters=[
                ('known_chip_z_in_world', 0.00417), # height (in mm) of the chip with respect to the breadboard
                ('camera_optical_frame_id', 'camera_color_optical_frame'),
                ('min_chip_area', 300),  # Reverted back to original
                ('max_chip_area', 1800),  # Reverted back to original
                ('epsilon_factor', 0.05),  # Reverted back to original
                ('expected_aspect_ratio', 2.0),
                ('aspect_ratio_tolerance', 0.7),  # Increased from 0.6 to allow minor variations
                ('show_debug_windows', False),
                ('tf_timeout', 1.0),
                ('max_processing_time', 0.1),  # Log warning if processing takes longer than this
                ('consistency_threshold', 0.4),  
                ('enable_consistency_fallback', True),  # Use best available poses if consistency fails
                ('process_interval', 0.2),  # Process every N seconds 
                ('history_size', 5),  # Increased from 3 to 5 frames for history
                ('max_detection_attempts', 6),  # Maximum detection attempts (6 default)
            ]
        )
        
        self.known_chip_z_in_world_ = self.get_parameter('known_chip_z_in_world').value
        self.camera_optical_frame_id_ = self.get_parameter('camera_optical_frame_id').value
        
        self.bridge = CvBridge()

        # TF2 Setup
        self.tf_buffer_ = tf2_ros.Buffer()
        self.tf_listener_ = tf2_ros.TransformListener(self.tf_buffer_, self, spin_thread=True)
        
        # Camera Intrinsics
        self.camera_matrix_ = None
        self.dist_coeffs_ = None
        self.fx_ = None
        self.fy_ = None
        self.cx_ = None
        self.cy_ = None

        # Stateful Detection Variables
        self.stable_poses_ = None
        self.detection_complete_ = False
        self.detection_attempts_ = 0
        self.max_detection_attempts_ = self.get_parameter('max_detection_attempts').value
        
        # Robustness Buffer
        self.detection_history_ = []
        self.history_size_ = self.get_parameter('history_size').value
        self.consistency_threshold_ = self.get_parameter('consistency_threshold').value

        # Rate Limiting
        self.last_process_time = 0
        self.process_interval = self.get_parameter('process_interval').value

        # Performance Monitoring
        self.max_processing_time = self.get_parameter('max_processing_time').value
        self.processing_times = []
        self.max_processing_history = 10

        # Publishers and Subscribers
        # Use sensor data QoS for image subscription
        qos_profile_image = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=DurabilityPolicy.VOLATILE
        )
        self.image_sub_ = self.create_subscription(
                                    Image,
                                    image_topic,
                                    self.image_callback,
                                    qos_profile_image)
        
        qos_profile_camera_info = QoSProfile(
            depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.VOLATILE)
        self.cam_info_sub_ = self.create_subscription(
                                    CameraInfo, 
                                    '/camera/camera/color/camera_info', 
                                    self.camera_info_callback, 
                                    qos_profile_camera_info)

        # Use transient local for pose data to ensure delivery
        qos_profile_poses = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,  # Keep more messages
            durability=DurabilityPolicy.TRANSIENT_LOCAL  # Survive publisher restarts
        )
        self.chip_world_pose_pub_ = self.create_publisher(
                                            PoseStampedArray, 
                                            '~/detected_chip_world_poses', 
                                            qos_profile=qos_profile_poses)
        
        # Service for refreshing detection
        self.refresh_detection_srv_ = self.create_service(
                                            Trigger, 
                                            '~/refresh_detection', 
                                            self.refresh_detection_callback)
        
        # Publish detection status
        self.status_pub_ = self.create_publisher(
            String, '~/detection_status', 1)

        self.get_logger().info("Chip Pose Calculator node started.")
        self.get_logger().info("Service '~/refresh_detection' available to trigger new detection.")

        self.debug_image_pub_ = self.create_publisher(Image, '~/debug_chip_image', 10)

        self.get_logger().info("Chip Pose Calculator node started.")
        
        # Debug flag - set to True to show all debug windows
        self.show_debug_windows = self.get_parameter('show_debug_windows').value
        
        # Create named windows if debug mode is enabled
        if self.show_debug_windows:
            cv2.namedWindow("RealSense Image with Detections", cv2.WINDOW_AUTOSIZE)
            cv2.namedWindow("Grayscale Image", cv2.WINDOW_AUTOSIZE)
            cv2.namedWindow("Debug View", cv2.WINDOW_AUTOSIZE)

    def camera_info_callback(self, msg: CameraInfo):
        if self.camera_matrix_ is None:
            self.camera_matrix_ = np.array(msg.k).reshape(3, 3)
            self.dist_coeffs_ = np.array(msg.d)
            self.fx_ = self.camera_matrix_[0, 0]
            self.fy_ = self.camera_matrix_[1, 1]
            self.cx_ = self.camera_matrix_[0, 2]
            self.cy_ = self.camera_matrix_[1, 2]
            self.get_logger().info("Camera intrinsics received.")
            # Unsubscribe after receiving camera info since it's static
            self.destroy_subscription(self.cam_info_sub_)
            self.cam_info_sub_ = None
    
    def refresh_detection_callback(self, request, response):
        """Service callback to refresh chip detection."""
        self.get_logger().info("Refresh detection requested. Resetting detection state.")
        
        # Reset detection state completely to force fresh detection
        self.detection_complete_ = False
        self.detection_attempts_ = 0
        self.detection_history_ = []
        self.stable_poses_ = None
        
        # Clear transform cache to force refresh
        # self.camera_transform_cache = None # Removed
        
        # Reset timing to ensure immediate processing
        self.last_process_time = 0.0
        
        # Publish empty poses immediately to clear any stale data
        empty_poses = PoseStampedArray()
        empty_poses.poses = []
        self.chip_world_pose_pub_.publish(empty_poses)
        
        # Publish status indicating fresh detection is needed
        status_msg = String()
        status_msg.data = "DETECTION_RESET"
        self.status_pub_.publish(status_msg)
        
        response.success = True
        response.message = "Detection state reset. New detection will begin on next image."
        
        return response
    
    def image_callback(self, msg: Image):
        start_time = time.time()
        
        current_time = self.get_clock().now().nanoseconds / 1e9
        
        # Check TF availability before processing
        try:
            if not self.tf_buffer_.can_transform(
                self.camera_optical_frame_id_, 'meca_base_link',
                rclpy.time.Time(seconds=0, nanoseconds=0)):
                self.get_logger().warn("TF transform not available yet, skipping frame")
                return
        except Exception as e:
            self.get_logger().warn(f"TF check failed: {e}")
            return
        
        # Always convert image for high-frequency debug publishing
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"Could not convert image: {e}")
            return
        
        # Always publish high-frequency debug image with most recent overlays
        self.publish_high_frequency_debug_image(cv_image, msg.header)
        
        # If detection is not complete, process immediately without rate limiting
        if not self.detection_complete_:
            # Force immediate processing for fresh detection
            self.last_process_time = 0.0
        else:
            # Only apply rate limiting when detection is complete and we're just republishing
            if current_time - self.last_process_time < self.process_interval:
                return  # Skip this frame
        self.last_process_time = current_time

        # If we have already found a stable set of poses, just republish them and exit.
        if self.detection_complete_ and self.stable_poses_ is not None:
            # Update the timestamp to show that the data is still "live"
            # even though the values are cached.
            # Note: PoseStampedArray doesn't have a header, so we don't update it
            self.chip_world_pose_pub_.publish(self.stable_poses_)
            
            # Publish status
            status_msg = String()
            status_msg.data = "DETECTION_COMPLETE"
            self.status_pub_.publish(status_msg)
            
            return  # Exit the callback early, saving CPU time.

        # If detection is not yet complete, we run the full logic ONCE.
        if not self.detection_complete_:
            self.detection_attempts_ += 1
            
            if self.fx_ is None:
                self.get_logger().warn("Waiting for camera intrinsics...")
                return

            if self.detection_attempts_ > self.max_detection_attempts_:
                self.get_logger().error(f"Failed to detect chips after {self.max_detection_attempts_} attempts. Publishing empty list.")
                self.detection_complete_ = True
                self.stable_poses_ = PoseStampedArray()
                # Note: PoseStampedArray doesn't have a header, so we don't set it
                self.chip_world_pose_pub_.publish(self.stable_poses_)
                
                # Publish status
                status_msg = String()
                status_msg.data = "DETECTION_FAILED"
                self.status_pub_.publish(status_msg)
                return

            self.get_logger().info(f"Detection attempt {self.detection_attempts_}/{self.max_detection_attempts_}: Processing image {msg.header.stamp.sec}.{msg.header.stamp.nanosec}")
            
            try:
                # Convert ROS Image message to OpenCV image
                cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            except Exception as e:
                self.get_logger().error(f"Could not convert image: {e}")
                return

            
            # Run multi-scale detection with multiple parameter combinations
            all_detections = self.detect_chips_multi_scale(cv_image)
            
            # Add diagnostic logging
            param_counts = {}
            for det in all_detections:
                key = f"block={det['block_size']}, kernel={det['kernel_size']}"
                param_counts[key] = param_counts.get(key, 0) + 1
            
            self.get_logger().info(f"Detection breakdown by parameters: {param_counts}")
            
            # Merge duplicate detections from different parameter combinations
            merged_detections = self.merge_detections(all_detections)
            
            # Filter false positives
            filtered_detections = self.filter_false_positives(merged_detections, cv_image.shape)
            
            self.get_logger().info(f"Multi-scale detection: {len(all_detections)} total detections, {len(merged_detections)} after merging, {len(filtered_detections)} after filtering")

            # Convert filtered detections to chip poses
            chip_pose_array_msg = PoseStampedArray()
            
            for det in filtered_detections:
                center_x_px, center_y_px = det['center']
                angle_deg = det['angle']
                
                # 3D Pose Calculation (Ray-Plane Intersection)
                time_stamp = msg.header.stamp
                
                # Convert pixel coordinates to normalized camera coordinates
                x_norm_cam = (center_x_px - self.cx_) / self.fx_
                y_norm_cam = (center_y_px - self.cy_) / self.fy_
                
                # Transform camera origin to base frame
                cam_origin_in_cam_frame = PointStamped()
                cam_origin_in_cam_frame.header.frame_id = self.camera_optical_frame_id_
                cam_origin_in_cam_frame.header.stamp = rclpy.time.Time(seconds=0, nanoseconds=0).to_msg()
                
                try:
                    cam_origin_in_base_frame = self.tf_buffer_.transform(cam_origin_in_cam_frame, 'meca_base_link', timeout=Duration(seconds=self.get_parameter('tf_timeout').value))
                except Exception as e:
                    self.get_logger().error(f"Failed to transform camera origin: {e}")
                    continue
                
                # Transform ray point to base frame
                ray_point_in_cam_frame = PointStamped()
                ray_point_in_cam_frame.header.frame_id = self.camera_optical_frame_id_
                ray_point_in_cam_frame.header.stamp = rclpy.time.Time(seconds=0, nanoseconds=0).to_msg()
                ray_point_in_cam_frame.point.x = x_norm_cam
                ray_point_in_cam_frame.point.y = y_norm_cam
                ray_point_in_cam_frame.point.z = 1.0

                try:
                    ray_point_in_base_frame = self.tf_buffer_.transform(ray_point_in_cam_frame, 'meca_base_link', timeout=Duration(seconds=self.get_parameter('tf_timeout').value))
                except Exception as e:
                    self.get_logger().error(f"Failed to transform ray point: {e}")
                    continue
                
                # Calculate ray-plane intersection
                Oc_base = np.array([cam_origin_in_base_frame.point.x, cam_origin_in_base_frame.point.y, cam_origin_in_base_frame.point.z])
                Pc_prime_base = np.array([ray_point_in_base_frame.point.x, ray_point_in_base_frame.point.y, ray_point_in_base_frame.point.z])
                ray_direction_base = Pc_prime_base - Oc_base
                
                if abs(ray_direction_base[2]) < 1e-9:  # Ray parallel to XY plane
                    self.get_logger().warn("Ray is parallel to the chip's XY plane in world frame.", throttle_duration_sec=5)
                    continue

                t_intersect = (self.known_chip_z_in_world_ - Oc_base[2]) / ray_direction_base[2]

                if t_intersect < 0:  # Intersection behind camera
                    self.get_logger().warn("Intersection with chip plane is behind the camera.", throttle_duration_sec=5)
                    continue

                chip_position_base = Oc_base + t_intersect * ray_direction_base
                
                # Calculate 3D orientation
                angle_rad_in_image = math.radians(angle_deg)
                rotation_in_cam = R.from_euler('z', angle_rad_in_image)
                quat_xyzw_cam = rotation_in_cam.as_quat()

                chip_orientation_in_cam = PoseStamped()
                chip_orientation_in_cam.header.stamp = rclpy.time.Time(seconds=0, nanoseconds=0).to_msg()
                chip_orientation_in_cam.header.frame_id = self.camera_optical_frame_id_
                chip_orientation_in_cam.pose.orientation.x = quat_xyzw_cam[0]
                chip_orientation_in_cam.pose.orientation.y = quat_xyzw_cam[1]
                chip_orientation_in_cam.pose.orientation.z = quat_xyzw_cam[2]
                chip_orientation_in_cam.pose.orientation.w = quat_xyzw_cam[3]

                # Transform orientation to base frame
                try:
                    chip_orientation_in_base = self.tf_buffer_.transform(chip_orientation_in_cam, 'meca_base_link', timeout=Duration(seconds=self.get_parameter('tf_timeout').value))
                except Exception as e:
                    self.get_logger().error(f"Failed to transform orientation: {e}")
                    continue
                
                # Orientation correction to ensure chip is flat with respect to base frame
                tilted_orientation = chip_orientation_in_base.pose.orientation
                R_tilted = R.from_quat([tilted_orientation.x,
                                       tilted_orientation.y,
                                       tilted_orientation.z,
                                       tilted_orientation.w])
                
                # Extract yaw angle and create corrected orientation
                euler_angles = R_tilted.as_euler('zyx', degrees=False)
                yaw_angle_rad = euler_angles[0]

                reflected_yaw_rad = -yaw_angle_rad
                final_yaw_rad = reflected_yaw_rad + math.radians(0.0)
                R_corrected = R.from_euler('z', final_yaw_rad, degrees=False)
                quat_corrected = R_corrected.as_quat()

                self.get_logger().info(f"Corrected orientation from yaw: {math.degrees(yaw_angle_rad):.2f} deg")
                
                # Create final chip pose
                final_chip_pose = PoseStamped()
                final_chip_pose.header.stamp = time_stamp
                final_chip_pose.header.frame_id = 'meca_base_link'
                
                final_chip_pose.pose.position.x = chip_position_base[0]
                final_chip_pose.pose.position.y = chip_position_base[1]
                final_chip_pose.pose.position.z = chip_position_base[2]
                
                final_chip_pose.pose.orientation.x = quat_corrected[0]
                final_chip_pose.pose.orientation.y = quat_corrected[1]
                final_chip_pose.pose.orientation.z = quat_corrected[2]
                final_chip_pose.pose.orientation.w = quat_corrected[3]
                
                chip_pose_array_msg.poses.append(final_chip_pose)
                self.get_logger().info(f"Successfully added chip pose to array. Total poses: {len(chip_pose_array_msg.poses)}")
                
        # Process detection history if detection is not complete
        if not self.detection_complete_:
            # Add this detection to history and check for consistency
            chip_count = len(chip_pose_array_msg.poses)
            self.detection_history_.append({
                'count': chip_count,
                'poses': chip_pose_array_msg.poses.copy() if chip_pose_array_msg.poses else []
            })
            
            # Keep only the last N frames
            if len(self.detection_history_) > self.history_size_:
                self.detection_history_.pop(0)
            
            self.get_logger().info(f"Detection attempt {self.detection_attempts_}: Found {chip_count} chips. History: {[d['count'] for d in self.detection_history_]}")
            
            # Check if we have enough history and consistent results
            if len(self.detection_history_) >= self.history_size_:
                chip_counts = [d['count'] for d in self.detection_history_]
                most_common_count = max(set(chip_counts), key=chip_counts.count)
                consistency_ratio = chip_counts.count(most_common_count) / len(chip_counts)
                
                # Check if we have any valid poses in the most common count
                poses_available = False
                for frame in self.detection_history_:
                    if frame['count'] == most_common_count and frame['poses']:
                        poses_available = True
                        break
                
                if consistency_ratio >= self.consistency_threshold_ and poses_available:
                    # Find the frame with the most common chip count and use its poses
                    best_frame = None
                    for frame in self.detection_history_:
                        if frame['count'] == most_common_count and frame['poses']:
                            best_frame = frame
                            break
                    
                    if best_frame and best_frame['poses']:
                        self.get_logger().info(f"Consistent detection achieved! {most_common_count} chips found in {consistency_ratio*100:.1f}% of frames. Caching stable poses.")
                        
                        # Create the stable message
                        self.stable_poses_ = PoseStampedArray()
                        self.stable_poses_.poses = best_frame['poses']
                        
                        self.detection_complete_ = True
                        
                        # Publish the stable list for the first time
                        self.chip_world_pose_pub_.publish(self.stable_poses_)
                        
                        # Publish status
                        status_msg = String()
                        status_msg.data = f"DETECTION_COMPLETE_{most_common_count}_chips"
                        self.status_pub_.publish(status_msg)
                    else:
                        self.get_logger().warn(f"Consistent count {most_common_count} but no valid poses found.")
                        # Publish status indicating the issue
                        status_msg = String()
                        status_msg.data = f"CONSISTENT_COUNT_{most_common_count}_NO_POSES"
                        self.status_pub_.publish(status_msg)
                else:
                    if not poses_available:
                        self.get_logger().warn(f"Inconsistent detection: {chip_counts}. Most common: {most_common_count} (no valid poses)")
                        status_msg = String()
                        status_msg.data = f"INCONSISTENT_NO_POSES_{most_common_count}"
                        self.status_pub_.publish(status_msg)
                    else:
                        self.get_logger().warn(f"Inconsistent detection: {chip_counts}. Need {self.consistency_threshold_*100:.1f}% consistency, got {consistency_ratio*100:.1f}%")
                        
                        # Fallback: use the best available poses if consistency fallback is enabled
                        if self.get_parameter('enable_consistency_fallback').value:
                            # Find the frame with the most chips that has valid poses
                            best_frame = None
                            max_chips = 0
                            for frame in self.detection_history_:
                                if frame['poses'] and frame['count'] > max_chips:
                                    best_frame = frame
                                    max_chips = frame['count']
                            
                            if best_frame and best_frame['poses']:
                                self.get_logger().info(f"Using consistency fallback: {max_chips} chips from best available frame")
                                
                                # Create the stable message
                                self.stable_poses_ = PoseStampedArray()
                                self.stable_poses_.poses = best_frame['poses']
                                
                                self.detection_complete_ = True
                                
                                # Publish the stable list
                                self.chip_world_pose_pub_.publish(self.stable_poses_)
                                
                                # Publish status
                                status_msg = String()
                                status_msg.data = f"DETECTION_FALLBACK_{max_chips}_chips"
                                self.status_pub_.publish(status_msg)
                            else:
                                status_msg = String()
                                status_msg.data = f"INCONSISTENT_{consistency_ratio*100:.0f}%_NO_FALLBACK"
                                self.status_pub_.publish(status_msg)
                        else:
                            status_msg = String()
                            status_msg.data = f"INCONSISTENT_{consistency_ratio*100:.0f}%"
                            self.status_pub_.publish(status_msg)
            else:
                self.get_logger().info(f"Building detection history: {len(self.detection_history_)}/{self.history_size_} frames")
                # Publish status indicating we're still building history
                status_msg = String()
                status_msg.data = f"BUILDING_HISTORY_{len(self.detection_history_)}_{self.history_size_}"
                self.status_pub_.publish(status_msg)
        
        # Performance monitoring
        processing_time = time.time() - start_time
        self.processing_times.append(processing_time)
        if len(self.processing_times) > self.max_processing_history:
            self.processing_times.pop(0)
        
        if processing_time > self.max_processing_time:
            avg_time = sum(self.processing_times) / len(self.processing_times)
            self.get_logger().warn(f"Slow processing: {processing_time:.3f}s (avg: {avg_time:.3f}s)")

    def detect_chips_multi_scale(self, cv_image, block_sizes=None, kernel_sizes=None):
        """
        Robust multi-scale chip detection that tries multiple parameter combinations
        and combines the results to handle different lighting/position conditions.
        """
        if block_sizes is None:
            block_sizes = self.DEFAULT_BLOCK_SIZES
        if kernel_sizes is None:
            kernel_sizes = self.DEFAULT_KERNEL_SIZES
            
        all_detections = []
        
        for block_size in block_sizes:
            for kernel_size in kernel_sizes:
                # Image preprocessing
                gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
                
                # Apply bilateral filter to reduce noise while preserving edges
                blurred = cv2.bilateralFilter(gray, self.BILATERAL_D, self.BILATERAL_SIGMA, self.BILATERAL_SIGMA)
                
                # Apply adaptive thresholding
                thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            cv2.THRESH_BINARY_INV, block_size, self.THRESH_C)

                # Apply morphological operations
                kernel = np.ones((3,3), np.uint8)
                thresh_closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
                thresh_opened = cv2.morphologyEx(thresh_closed, cv2.MORPH_OPEN, kernel)
                
                # Additional closing to handle reflection gaps
                kernel_close = np.ones((kernel_size,kernel_size), np.uint8)
                thresh_final = cv2.morphologyEx(thresh_opened, cv2.MORPH_CLOSE, kernel_close)
                
                # Find contours
                contours, hierarchy = cv2.findContours(thresh_final, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # Filter out large contours
                filtered_contours = []
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if area < self.LARGE_CONTOUR_THRESHOLD:
                        filtered_contours.append(contour)
                
                contours = filtered_contours
                
                # Chip detection and validation
                min_chip_area = self.get_parameter('min_chip_area').value
                max_chip_area = self.get_parameter('max_chip_area').value
                epsilon_factor = self.get_parameter('epsilon_factor').value
                expected_aspect_ratio = self.get_parameter('expected_aspect_ratio').value
                aspect_ratio_tolerance = self.get_parameter('aspect_ratio_tolerance').value

                # Process each contour
                for contour in contours:
                    area = cv2.contourArea(contour)

                    # Area filter
                    if not (min_chip_area < area < max_chip_area):
                        continue
                    
                    # Shape approximation
                    peri = cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, epsilon_factor * peri, True)

                    # Must have 4 vertices
                    if len(approx) != 4:
                        continue
                    
                    # Get bounding rectangle
                    rect = cv2.minAreaRect(contour)
                    (center_x_px, center_y_px) = rect[0]
                    (w_px, h_px) = rect[1]
                    angle_deg = rect[2]

                    # Handle orientation
                    if w_px < h_px:
                        angle_deg = angle_deg + 90.0
                        actual_width_px = h_px
                        actual_height_px = w_px
                    else:
                        angle_deg = angle_deg
                        actual_width_px = w_px
                        actual_height_px = h_px
                    
                    # Normalize angle
                    if angle_deg > 90.0:
                        angle_deg -= 180.0
                    if angle_deg < -90.0:
                        angle_deg += 180.0
                        
                    if actual_width_px <= 0 or actual_height_px <= 0:
                        continue

                    # Aspect ratio check
                    current_aspect_ratio = actual_width_px / actual_height_px
                    if not (expected_aspect_ratio - aspect_ratio_tolerance < current_aspect_ratio < expected_aspect_ratio + aspect_ratio_tolerance):
                        continue
                    
                    # Convexity check
                    if not cv2.isContourConvex(approx):
                        continue
                    
                    # Valid chip detected
                    detection = {
                        'center': (center_x_px, center_y_px),
                        'area': area,
                        'angle': angle_deg,
                        'aspect_ratio': current_aspect_ratio,
                        'block_size': block_size,
                        'kernel_size': kernel_size,
                        'contour': contour
                    }
                    all_detections.append(detection)
        
        return all_detections

    def merge_detections(self, detections, distance_threshold=None):
        """
        Merge duplicate detections from different parameter combinations.
        Uses clustering to group nearby detections and select the best one from each group.
        """
        if distance_threshold is None:
            distance_threshold = self.MERGE_DISTANCE_THRESHOLD
        if not detections:
            return []
        
        # Group detections by proximity
        merged = []
        used = set()
        
        for i, det1 in enumerate(detections):
            if i in used:
                continue
                
            # Find all detections close to this one
            group = [det1]
            used.add(i)
            
            for j, det2 in enumerate(detections):
                if j in used:
                    continue
                    
                # Calculate distance between centers
                center1 = det1['center']
                center2 = det2['center']
                distance = ((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)**0.5
                
                if distance < distance_threshold:
                    group.append(det2)
                    used.add(j)
            
            # Select the best detection from this group
            # Prefer detections with better aspect ratio (closer to 2.0) and higher confidence
            best_detection = min(group, key=lambda x: abs(x['aspect_ratio'] - 2.0))
            
            # Add confidence score based on how many parameter combinations found this detection
            best_detection['confidence'] = len(group)
            merged.append(best_detection)
        
        return merged

    def filter_false_positives(self, detections, image_shape):
        """
        Filter out likely false positives based on various criteria.
        """
        filtered = []
        height, width = image_shape[:2]
        
        # Get parameters for filtering
        min_chip_area = self.get_parameter('min_chip_area').value
        max_chip_area = self.get_parameter('max_chip_area').value
        expected_aspect_ratio = self.get_parameter('expected_aspect_ratio').value
        aspect_ratio_tolerance = self.get_parameter('aspect_ratio_tolerance').value
        
        for det in detections:
            center_x, center_y = det['center']
            
            # Filter 1: Aspect ratio
            if not (expected_aspect_ratio - aspect_ratio_tolerance < det['aspect_ratio'] < expected_aspect_ratio + aspect_ratio_tolerance):
                continue
                
            # Filter 2: Area bounds
            if not (min_chip_area < det['area'] < max_chip_area):
                continue
                
            # Filter 3: Position filtering - exclude near edges
            edge_margin = 50  # pixels from edge
            if (center_x < edge_margin or center_x > width - edge_margin or 
                center_y < edge_margin or center_y > height - edge_margin):
                continue
                
            # Filter 4: Confidence filtering
            if det['confidence'] < self.CONFIDENCE_THRESHOLD:
                continue
                
            filtered.append(det)
        
        return filtered

    def publish_high_frequency_debug_image(self, cv_image, header):
        """Publish high-frequency debug image with live processing of current frame."""
        try:
            # Always process the current frame live for truly live debug images
            self.process_and_publish_live_debug_image(cv_image, header)
            
        except Exception as e:
            self.get_logger().error(f"Error publishing high-frequency debug image: {e}")

    def process_and_publish_live_debug_image(self, cv_image, header):
        """Process image and publish a live debug image with current detection overlays."""
        try:
            # Multi-scale detection with multiple parameter combinations
            all_detections = self.detect_chips_multi_scale(cv_image)
            
            # Add diagnostic logging
            param_counts = {}
            for det in all_detections:
                key = f"block={det['block_size']}, kernel={det['kernel_size']}"
                param_counts[key] = param_counts.get(key, 0) + 1
            
            self.get_logger().info(f"Detection breakdown by parameters: {param_counts}")
            
            # Merge duplicate detections from different parameter combinations
            merged_detections = self.merge_detections(all_detections)
            
            # Filter false positives
            filtered_detections = self.filter_false_positives(merged_detections, cv_image.shape)
            
            # Create debug image
            debug_img = cv_image.copy()
            
            # Draw all filtered detections
            for i, det in enumerate(filtered_detections):
                # Get contour points for bounding box
                rect = cv2.minAreaRect(det['contour'])
                box_points_float = cv2.boxPoints(rect)
                box_points_int = np.intp(box_points_float)
                
                # Draw green bounding box
                cv2.drawContours(debug_img, [box_points_int], 0, (0, 255, 0), 2)
                
                # Draw red center point
                center_x, center_y = det['center']
                cv2.circle(debug_img, (int(center_x), int(center_y)), 5, (0, 0, 255), -1)
                
                # Draw chip number
                cv2.putText(debug_img, str(i+1), (int(center_x-10), int(center_y-10)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Add text overlay with detection count
            cv2.putText(debug_img, f"Chips detected: {len(filtered_detections)}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Publish debug image
            self.publish_debug_image(debug_img, header)

        except Exception as e:
            self.get_logger().error(f"Error in process_and_publish_live_debug_image: {e}")

    def publish_debug_image(self, output_contours_image, header):
        """Method to publish debug image"""
        try:
            # Convert OpenCV image to ROS Image message
            debug_img_msg = self.bridge.cv2_to_imgmsg(output_contours_image, encoding="bgr8")

            # Copy header to keep timestamp and frame_id consistent
            debug_img_msg.header = header

            self.debug_image_pub_.publish(debug_img_msg)

        except Exception as e:
            self.get_logger().error(f"Error publishing debug image: {e}")

    def cleanup(self):
        """Clean up resources before shutdown."""
        try:
            # Clear TF2 buffer
            if hasattr(self, 'tf_buffer_'):
                self.tf_buffer_.clear()
            
            # Close OpenCV windows
            cv2.destroyAllWindows()
            
            # Clear caches
            self.detection_history_.clear()
            
            self.get_logger().info("Cleanup completed.")
        except Exception as e:
            self.get_logger().error(f"Error during cleanup: {e}")

    def __del__(self):
        """Destructor to ensure cleanup."""
        self.cleanup()

def main(rclpy_args=None):
    rclpy.init(args=rclpy_args)

    image_subscriber = chip_detection() # Uses default topic

    try:
        rclpy.spin(image_subscriber)
    except KeyboardInterrupt:
        image_subscriber.get_logger().info("Keyboard interrupt, shutting down.")
    except Exception as e:
        image_subscriber.get_logger().error(f"Unexpected error: {e}")
    finally:
        # Clean up resources
        image_subscriber.cleanup()
        
        # Destroy the node explicitly
        image_subscriber.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()