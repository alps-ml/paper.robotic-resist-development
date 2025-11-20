#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge # Package to convert ROS images to OpenCV
import cv2
import numpy as np 
from geometry_msgs.msg import PointStamped
from std_msgs.msg import Float32

class chip_detection(Node):
    def __init__(self, image_topic="/camera/camera/color/image_rect_raw"):
        super().__init__('chip_detection')
        self.subscription = self.create_subscription(
            Image,
            image_topic,
            self.image_callback,
            10) # QoS profile depth
        self.subscription  # prevent unused variable warning
        self.bridge = CvBridge()
        self.get_logger().info(f"Subscribed to {image_topic}")

        # --- Create Publishers ---
        self.chip_center_publisher = self.create_publisher(PointStamped, 'detected_chip_center_pixels', 10)
        self.chip_angle_publisher = self.create_publisher(Float32, 'detected_chip_angle_pixels', 10)
        
        self.camera_optical_frame_id = "camera_color_optical_frame"  #to find this name: ros2 topic echo /tf_static 
        
        # Debug flag - set to True to show all debug windows
        self.show_debug_windows = False
        
        # Define window names
        self.window_main = "RealSense Image with Detections"
        self.window_gray = "Grayscale Image"
        self.window_edges = "Canny Edges"
        self.window_debug = "Debug View"
        
        # Create named windows
        cv2.namedWindow(self.window_main, cv2.WINDOW_AUTOSIZE)
        if self.show_debug_windows:
            cv2.namedWindow(self.window_gray, cv2.WINDOW_AUTOSIZE)
            #cv2.namedWindow(self.window_edges, cv2.WINDOW_AUTOSIZE)
            cv2.namedWindow(self.window_debug, cv2.WINDOW_AUTOSIZE)

    def image_callback(self, msg):
        self.get_logger().info(f"Received image: {msg.header.stamp.sec}.{msg.header.stamp.nanosec}")
        try:
            # Convert ROS Image message to OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"Could not convert image: {e}")
            return

        # --- Enhanced Preprocessing ---
        # 1. Convert to grayscale
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        
        # 2. Apply adaptive histogram equalization to improve contrast
        #clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        #gray = clahe.apply(gray)
        # 3. Apply bilateral filter to reduce noise while preserving edges
        blurred = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # 4. Adaptive Thresholding (Alternative to Canny for findContours input)
        # Experiment with blockSize and C
        block_size = 11
        C_val = 2
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY_INV, block_size, C_val)
        # Morphological operations to clean up the thresholded image
        kernel = np.ones((3,3), np.uint8)
        thresh_closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh_opened = cv2.morphologyEx(thresh_closed, cv2.MORPH_OPEN, kernel)
        
        # 6. Edge detection with adjusted parameters
        # edges = cv2.Canny(blurred, 30, 100)  # Lower thresholds for better edge detection
        
        # Display preprocessing results
        if self.show_debug_windows:
            cv2.imshow(self.window_gray, gray)
            #cv2.imshow(self.window_edges, edges)
            cv2.imshow("Adaptive Threshold", thresh_opened)
        
        # Create debug image
        debug_img = cv_image.copy()
        
        try:
            contours, hierarchy = cv2.findContours(thresh_opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        except Exception as e:
            self.get_logger().error(f"Error finding contours: {e}")
            return

        # Draw all contours in debug image
        cv2.drawContours(debug_img, contours, -1, (0, 255, 255), 1)  # Yellow for all contours

        # Add contour count to debug image
        cv2.putText(debug_img, f"Total contours: {len(contours)}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        output_contours_image = cv_image.copy() # For drawing final candidates

        # Contour processing loop:
        min_chip_area = 300  # tune this
        max_chip_area = 800  # e.g., if chip area is ~ 600 pixels^2
        epsilon_factor = 0.05 # START TUNING THIS (0.01 to 0.04)
        expected_aspect_ratio = 2.6 # for a 10x3.4mm chip
        aspect_ratio_tolerance = 0.6

        chips_found_this_run = 0
        for contour in contours:
            try:
                area = cv2.contourArea(contour)

                if min_chip_area < area < max_chip_area:
                    peri = cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, epsilon_factor * peri, True)

                    if len(approx) == 4:
                        rect = cv2.minAreaRect(contour)
                        box_points_float = cv2.boxPoints(rect)
                        box_points_int = np.intp(box_points_float)

                        (center_x_px, center_y_px) = rect[0]
                        (w_px, h_px) = rect[1]
                        angle_deg = rect[2]

                        if w_px <= 0 or h_px <= 0:
                            continue

                        current_aspect_ratio = max(w_px, h_px) / min(w_px, h_px)

                        if not (expected_aspect_ratio - aspect_ratio_tolerance < current_aspect_ratio < expected_aspect_ratio + aspect_ratio_tolerance):
                            continue # Failed aspect ratio test

                        #self.get_logger().info(
                        #    f"PRE-CONVEXITY: Area={area:.2f}, AR={current_aspect_ratio:.2f}, "
                        #    f"NumApproxPts={len(approx)}"
                        #)

                        if cv2.isContourConvex(approx):
                            print(f"GOOD Candidate: Center=({int(center_x_px)}, {int(center_y_px)}), "
                                f"Angle={angle_deg:.2f}, Area={area:.2f}, "
                                f"Size={w_px:.1f}x{h_px:.1f}, AR={current_aspect_ratio:.2f}")
                            cv2.drawContours(output_contours_image, [box_points_int], 0, (0, 255, 0), 2)  # Green box
                            cv2.circle(output_contours_image, (int(center_x_px), int(center_y_px)), 5, (0, 0, 255), -1)  # Red center

                        # --- Publish the Detection ---
                            now = self.get_clock().now().to_msg() # Or use msg.header.stamp from the input image

                            # Publish center point (pixel coordinates)
                            center_msg = PointStamped()
                            center_msg.header.stamp = now
                            center_msg.header.frame_id = self.camera_optical_frame_id
                            center_msg.point.x = float(center_x_px)
                            center_msg.point.y = float(center_y_px)
                            center_msg.point.z = 0.0 # Pixel coordinates are 2D, z is not applicable here
                            self.chip_center_publisher.publish(center_msg)

                            # Publish angle (degrees, from minAreaRect)
                            angle_msg = Float32()
                            angle_msg.data = float(angle_deg)
                            self.chip_angle_publisher.publish(angle_msg)
                

                            chips_found_this_run += 1
            except Exception as e:
                self.get_logger().error(f"Error processing contour: {e}")
                continue
        
        # Display results
        try:
            cv2.imshow(self.window_main, output_contours_image)  # Show the image with detections
            cv2.waitKey(1)
        except Exception as e:
            self.get_logger().error(f"Error displaying image: {e}")

    def __del__(self):
        """Cleanup when the node is destroyed"""
        cv2.destroyWindow(self.window_main)
        if self.show_debug_windows:
            cv2.destroyWindow(self.window_gray)
            #cv2.destroyWindow(self.window_edges)
            cv2.destroyWindow(self.window_debug)
            cv2.destroyWindow("Adaptive Threshold")

def main(rclpy_args=None):
    rclpy.init(args=rclpy_args)

    # You can pass the topic name as a command-line argument or change the default
    # For example, to subscribe to the aligned depth image:
    # image_subscriber = ImageSubscriber(image_topic="/camera/aligned_depth_to_color/image_raw")
    image_subscriber = chip_detection() # Uses default topic

    try:
        rclpy.spin(image_subscriber)
    except KeyboardInterrupt:
        image_subscriber.get_logger().info("Keyboard interrupt, shutting down.")
    finally:
        # Destroy the node explicitly
        # (optional - otherwise it will be done automatically
        # when the garbage collector destroys the node object)
        image_subscriber.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        cv2.destroyAllWindows() # Close OpenCV windows

if __name__ == '__main__':
    main()