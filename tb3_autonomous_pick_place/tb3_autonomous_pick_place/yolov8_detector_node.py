import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose
from std_msgs.msg import Header, String, Bool
from geometry_msgs.msg import Pose2D
from cv_bridge import CvBridge
import cv2
import numpy as np
from ultralytics import YOLO
import torch
import os
import time
from rclpy.executors import MultiThreadedExecutor
from ament_index_python.packages import get_package_share_directory

class YOLOv8Detector(Node):
    def __init__(self):
        super().__init__('yolov8_detector')
        
        # Detection state variables
        self.detection_active = False
        self.target_color = None
        self.object_detected = False
        self.detection_published = False
        self.valid_colors = ['red', 'blue', 'white']
        
        # Timeout variables
        self.detection_start_time = None
        self.detection_timeout = 90.0  # seconds to wait before declaring "not detected"
        self.timeout_timer = None
        
        # Parameters
        package_dir = get_package_share_directory('tb3_autonomous_pick_place')
        model_path = os.path.join(package_dir, 'models', 'yolov8_custom.pt')

        self.declare_parameter('model_path', model_path)
        self.declare_parameter('confidence_threshold', 0.5)
        self.declare_parameter('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.declare_parameter('input_topic', '/image_raw')
        self.declare_parameter('output_topic', '/pickup_detection_bbox')
        self.declare_parameter('detection_timeout', 90.0) 
        
        # Get parameters
        model_path = self.get_parameter('model_path').get_parameter_value().string_value
        self.confidence_threshold = self.get_parameter('confidence_threshold').get_parameter_value().double_value
        device = self.get_parameter('device').get_parameter_value().string_value
        input_topic = self.get_parameter('input_topic').get_parameter_value().string_value
        output_topic = self.get_parameter('output_topic').get_parameter_value().string_value
        self.detection_timeout = self.get_parameter('detection_timeout').get_parameter_value().double_value
        
        # Load YOLO model
        try:
            self.model = YOLO(model_path)
            self.model.to(device)
            self.get_logger().info(f'YOLO model loaded from {model_path} on {device}')
        except Exception as e:
            self.get_logger().error(f'Failed to load YOLO model: {e}')
            return
        
        # Class mapping 
        self.class_mapping = {
            'blue cylinder horizontal': 'blue',
            'blue cylinder vertical': 'blue',
            'red cylinder horizontal': 'red',
            'red cylinder vertical': 'red',
            'white cylinder horizontal': 'white',
            'white cylinder vertical': 'white'
        }
        
        # CV Bridge for image conversion
        self.bridge = CvBridge()
        
        # Subscribers
        self.image_subscriber = self.create_subscription(
            Image,
            input_topic,
            self.image_callback,
            10
        )
        
        # Subscribe to pickup position reached flag
        self.pickup_flag_subscriber = self.create_subscription(
            Bool,
            '/exploration_complete',
            self.pickup_flag_callback,
            10
        )
        
        # Subscribe to target color command
        self.target_color_subscriber = self.create_subscription(
            String,
            '/target_color',
            self.target_color_callback,
            10
        )
        
        # Publishers
        self.pickup_bbox_publisher = self.create_publisher(
            Detection2DArray,
            '/pickup_detection_bbox',
            10
        )
        
        # Publisher for detection status (success/failure)
        self.detection_status_publisher = self.create_publisher(
            String,
            '/detection_status',
            10
        )
        
        # Publish annotated image for visualization
        self.annotated_image_publisher = self.create_publisher(
            Image,
            '/detections/annotated_image',
            10
        )

        
        
        # Timer to check subscriber count
        self.subscriber_check_timer = self.create_timer(0.5, self.check_subscribers)
        
        self.get_logger().info('YOLO Cylinder Detector Node initialized')
        self.get_logger().info(f'Valid colors for detection: {self.valid_colors}')
        self.get_logger().info(f'Detection timeout set to: {self.detection_timeout} seconds')
        
    def pickup_flag_callback(self, msg):
        """Callback for pickup position reached flag"""
        if msg.data:
            self.get_logger().info("Pickup position reached flag received")
            # Reset all detection states when position is reached
            self.reset_detection_state()
        
    def target_color_callback(self, msg):
        """Callback for target color command"""
        requested_color = msg.data.lower().strip()
        
        if requested_color not in self.valid_colors:
            self.get_logger().error(f"Invalid color '{requested_color}'. Valid colors are: {self.valid_colors}")
            return
        
        # Set target color and activate detection
        self.target_color = requested_color
        self.detection_active = True
        self.object_detected = False
        self.detection_published = False
        self.detection_start_time = time.time()  # Record start time
        
        # Start timeout timer
        if self.timeout_timer:
            self.timeout_timer.cancel()
        self.timeout_timer = self.create_timer(self.detection_timeout, self.detection_timeout_callback)
        
        self.get_logger().info(f"Target color set to: {self.target_color}")
        self.get_logger().info(f"Detection activated - searching for target object (timeout: {self.detection_timeout}s)")
        
    def detection_timeout_callback(self):
        """Called when detection timeout expires"""
        if self.detection_active and not self.object_detected and not self.detection_published:
            # Target not detected within timeout period
            self.get_logger().warn(f"Target '{self.target_color}' not detected within {self.detection_timeout} seconds")
            
            # Publish failure status
            status_msg = String()
            status_msg.data = f"Target {self.target_color} not detected with confidence {self.confidence_threshold} after {self.detection_timeout} seconds"
            self.detection_status_publisher.publish(status_msg)
            
            # Log the failure message
            self.get_logger().error(f"Target {self.target_color} not detected with confidence {self.confidence_threshold}")
            
            # Reset detection state
            self.reset_detection_state()
        
        # Cancel the timer since it's a one-shot
        if self.timeout_timer:
            self.timeout_timer.cancel()
            self.timeout_timer = None
    
    def reset_detection_state(self):
        """Reset detection state for next cycle"""
        self.detection_active = False
        self.target_color = None
        self.object_detected = False
        self.detection_published = False
        self.detection_start_time = None
        
        # Cancel timeout timer if active
        if self.timeout_timer:
            self.timeout_timer.cancel()
            self.timeout_timer = None
            
        self.get_logger().info("Detection state reset - ready for next target")
        
    def check_subscribers(self):
        """Check if sensor fusion node is subscribed to our topic"""
        subscriber_count = self.pickup_bbox_publisher.get_subscription_count()
        return subscriber_count > 0
        
    def image_callback(self, msg):
        """Process incoming images when detection is active"""
        # Only process if detection is active, object not yet detected, and not already published
        if not self.detection_active or self.object_detected or self.detection_published or self.target_color is None:
            return
            
        try:
            # Convert ROS Image to OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
            # Run YOLO inference
            results = self.model(cv_image, conf=self.confidence_threshold)
            
            # Process detections for target color
            detection_found = self.process_target_detections(results[0], msg.header)
            
            # Optionally publish annotated image
            if self.detection_active:
                annotated_image = self.draw_detections(cv_image, results[0])
                if annotated_image is not None:
                    annotated_msg = self.bridge.cv2_to_imgmsg(annotated_image, encoding='bgr8')
                    annotated_msg.header = msg.header
                    self.annotated_image_publisher.publish(annotated_msg)
                    
        except Exception as e:
            self.get_logger().error(f'Error processing image: {e}')
    
    def process_target_detections(self, result, header):
        """Process detections and publish only target color objects"""
        if result.boxes is None:
            return False
        
        # Check if already published
        if self.detection_published:
            return False
        
        target_detections = []
        
        # Filter detections for target color
        for box in result.boxes:
            # Get box coordinates (x1, y1, x2, y2)
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            
            # Get confidence and class
            confidence = float(box.conf[0].cpu().numpy())
            class_id = int(box.cls[0].cpu().numpy())
            
            # Get class name from model
            class_name = self.model.names[class_id]
            
            # Map to simplified color name
            simplified_class = self.class_mapping.get(class_name, class_name)
            
            # Check if this is our target color
            if simplified_class == self.target_color:
                detection_info = {
                    'box': box,
                    'confidence': confidence,
                    'class_name': simplified_class,
                    'coordinates': (x1, y1, x2, y2),
                    'center': ((x1 + x2) / 2, (y1 + y2) / 2)
                }
                target_detections.append(detection_info)
        
        if target_detections:
            # Select the best detection based on your criteria
            best_detection = self.select_best_detection(target_detections)
            
            # Check if sensor fusion node is subscribed
            if self.check_subscribers():
                # Create and publish detection message
                detection_array = self.create_detection_message(best_detection, header)
                self.pickup_bbox_publisher.publish(detection_array)
                
                # Log detection
                center_x, center_y = best_detection['center']
                self.get_logger().info(
                    f"Object with {self.target_color} detected at ({center_x:.1f}, {center_y:.1f}) "
                    f"coordinates with confidence: {best_detection['confidence']:.2f}"
                )
                
                # Publish success status
                status_msg = String()
                status_msg.data = f"Target {self.target_color} detected successfully with confidence {best_detection['confidence']:.2f}"
                self.detection_status_publisher.publish(status_msg)
                
                # Set flags to prevent further detections
                self.object_detected = True
                self.detection_published = True
                self.detection_active = False
                
                # Cancel timeout timer since we found the target
                if self.timeout_timer:
                    self.timeout_timer.cancel()
                    self.timeout_timer = None
                
                self.get_logger().info("Detection published once - stopping detection process")
                self.reset_detection_state()
                return True
            else:
                self.get_logger().warn("Sensor fusion node not subscribed - waiting for subscriber")
        
        return False
    
    def select_best_detection(self, detections):
        """Select the best detection based on criteria"""
        if len(detections) == 1:
            return detections[0]
        
        # If multiple objects with same color, select based on:
        # 1. Higher position (lower y-coordinate for top object)
        # 2. If similar height, select closest (largest bounding box area as approximation)
        
        # Sort by y-coordinate (top objects have lower y values)
        detections.sort(key=lambda d: d['center'][1])
        
        # Check if top objects are at similar height (within threshold)
        top_detection = detections[0]
        similar_height_detections = [top_detection]
        
        height_threshold = 50  # pixels
        for detection in detections[1:]:
            if abs(detection['center'][1] - top_detection['center'][1]) < height_threshold:
                similar_height_detections.append(detection)
            else:
                break
        
        # If multiple objects at similar height, select the one with largest area (closest)
        if len(similar_height_detections) > 1:
            def calculate_area(detection):
                x1, y1, x2, y2 = detection['coordinates']
                return (x2 - x1) * (y2 - y1)
            
            similar_height_detections.sort(key=calculate_area, reverse=True)
            selected = similar_height_detections[0]
            self.get_logger().info("Multiple objects at similar height - selected closest (largest area)")
        else:
            selected = top_detection
            self.get_logger().info("Selected top object")
        
        return selected
    
    def create_detection_message(self, detection_info, header):
        """Create Detection2DArray message from detection info"""
        detection_array = Detection2DArray()
        detection_array.header = header
        
        # Create Detection2D message
        detection = Detection2D()
        detection.header = header
        
        # Set bounding box
        x1, y1, x2, y2 = detection_info['coordinates']
        detection.bbox.center.position.x = float((x1 + x2) / 2)
        detection.bbox.center.position.y = float((y1 + y2) / 2)
        detection.bbox.center.theta = 0.0
        detection.bbox.size_x = float(x2 - x1)
        detection.bbox.size_y = float(y2 - y1)
        
        # Set hypothesis (class and confidence)
        hypothesis = ObjectHypothesisWithPose()
        hypothesis.hypothesis.class_id = detection_info['class_name']
        hypothesis.hypothesis.score = detection_info['confidence']
        
        detection.results.append(hypothesis)
        detection_array.detections.append(detection)
        
        return detection_array
    
    def draw_detections(self, image, result):
        """Draw bounding boxes and labels on image for visualization"""
        if result.boxes is None:
            return image
        
        annotated_image = image.copy()
        
        # Define colors for each cylinder type
        colors = {
            'blue': (255, 0, 0),    # BGR format
            'red': (0, 0, 255),
            'white': (255, 255, 255)
        }
        
        for box in result.boxes:
            # Get box coordinates
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            
            # Get confidence and class
            confidence = float(box.conf[0].cpu().numpy())
            class_id = int(box.cls[0].cpu().numpy())
            class_name = self.model.names[class_id]
            simplified_class = self.class_mapping.get(class_name, class_name)
            
            # Only draw if detection is active and matches target color
            if self.detection_active and simplified_class == self.target_color:
                # Get color for this class
                color = colors.get(simplified_class, (0, 255, 0))
                
                # Draw bounding box
                cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)
                
                # Draw label
                label = f"{simplified_class}: {confidence:.2f}"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                cv2.rectangle(annotated_image, (x1, y1 - label_size[1] - 10), 
                             (x1 + label_size[0], y1), color, -1)
                cv2.putText(annotated_image, label, (x1, y1 - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        return annotated_image


def main(args=None):
    rclpy.init(args=args)

    node = YOLOv8Detector()
    node.get_logger().info('YOLOv8 Detector Node is running')
    node.get_logger().info('Waiting for pickup position reached flag and target color...')

    executor = MultiThreadedExecutor(num_threads=6)  
    executor.add_node(node)

    try:
        executor.spin()
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
