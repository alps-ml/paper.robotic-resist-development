#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge # Package to convert ROS images to OpenCV
import cv2

class ImageSubscriber(Node):
    def __init__(self, image_topic="/camera/camera/color/image_rect_raw"):
        super().__init__('realsense_image_subscriber')
        self.subscription = self.create_subscription(
            Image,
            image_topic,
            self.image_callback,
            10) # QoS profile depth
        self.subscription  # prevent unused variable warning
        self.bridge = CvBridge()
        self.get_logger().info(f"Subscribed to {image_topic}")

    def image_callback(self, msg):
        self.get_logger().info(f"Received image: {msg.header.stamp.sec}.{msg.header.stamp.nanosec}")
        try:
            # Convert ROS Image message to OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"Could not convert image: {e}")
            return

        # --- Your image processing code starts here ---
        # For example, display the image:
        cv2.imshow("RealSense Image", cv_image)
        cv2.waitKey(1) # Process GUI events and wait 1ms

        # Example: Print image shape
        # self.get_logger().info(f"Image shape: {cv_image.shape}")

        # If you need to access pixel data:
        # (x, y) = (100, 200) # example pixel coordinates
        # b, g, r = cv_image[y, x]
        # self.get_logger().info(f"Pixel BGR at ({x},{y}): ({b},{g},{r})")

        # --- Your image processing code ends here ---

def main(rclpy_args=None):
    rclpy.init(args=rclpy_args)

    # You can pass the topic name as a command-line argument or change the default
    # For example, to subscribe to the aligned depth image:
    # image_subscriber = ImageSubscriber(image_topic="/camera/aligned_depth_to_color/image_raw")
    image_subscriber = ImageSubscriber() # Uses default topic

    try:
        rclpy.spin(image_subscriber)
    except KeyboardInterrupt:
        self.get_logger().info("Keyboard interrupt, shutting down.")
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