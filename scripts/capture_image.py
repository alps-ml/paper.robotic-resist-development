#!/usr/bin/env python3

# This script captures a single image from a specified ROS image topic and saves it to a file.

import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import time
import argparse

class ImageCaptureNode(Node):
    def __init__(self, image_topic="/camera/camera/color/image_rect_raw"):
        super().__init__('image_capture_node')
        
        self.bridge = CvBridge()
        self.latest_image = None
        self.image_received = False
        
        # Subscribe to image topic
        self.image_sub = self.create_subscription(
            Image,
            image_topic,
            self.image_callback,
            10
        )
        
        self.get_logger().info(f"Subscribed to {image_topic}")
    
    def image_callback(self, msg):
        try:
            self.latest_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.image_received = True
            self.get_logger().info(f"Received image: {self.latest_image.shape}")
        except Exception as e:
            self.get_logger().error(f"Error converting image: {e}")

def main():
    parser = argparse.ArgumentParser(description='Capture one image from ROS and save it')
    parser.add_argument('--topic', default='/camera/camera/color/image_rect_raw', 
                       help='ROS image topic to subscribe to')
    parser.add_argument('--output', default='captured_image.jpg', 
                       help='Output filename for captured image')
    parser.add_argument('--timeout', type=float, default=5.0, 
                       help='Timeout in seconds to wait for image')
    
    args = parser.parse_args()
    
    # Initialize ROS
    rclpy.init()
    
    # Create node and wait for image
    node = ImageCaptureNode(args.topic)
    
    print(f"Waiting up to {args.timeout} seconds for image from {args.topic}...")
    start_time = time.time()
    
    while rclpy.ok() and not node.image_received and (time.time() - start_time) < args.timeout:
        rclpy.spin_once(node, timeout_sec=0.1)
        print(".", end="", flush=True)
    
    print()  # New line
    
    if not node.image_received:
        print(f"Timeout: No image received from {args.topic}")
        node.destroy_node()
        rclpy.shutdown()
        return False
    
    # Save the captured image
    cv2.imwrite(args.output, node.latest_image)
    print(f"Image saved to: {args.output}")
    
    # Cleanup
    node.destroy_node()
    rclpy.shutdown()
    return True

if __name__ == "__main__":
    main()