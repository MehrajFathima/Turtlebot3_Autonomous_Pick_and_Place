import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from sensor_msgs.msg import LaserScan, Image
from vision_msgs.msg import Detection2DArray
from geometry_msgs.msg import PoseArray, Pose, Point, Quaternion
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Header, ColorRGBA
import tf2_ros
import tf2_geometry_msgs
from tf2_ros import TransformException
import numpy as np
import math
from scipy.spatial.transform import Rotation as R
import message_filters
from rclpy.executors import MultiThreadedExecutor
from builtin_interfaces.msg import Time
import time
import cv2

class SensorFusionNode(Node):
    def __init__(self):
        super().__init__('sensor_fusion_node')
        
        # Parameters
        self.declare_parameter('transform_timeout', 0.1)
        self.declare_parameter('sync_tolerance', 5.0)
        self.declare_parameter('min_detection_confidence', 0.5)
        self.declare_parameter('max_lidar_range', 3.5)
        self.declare_parameter('min_lidar_range', 0.12)
        self.declare_parameter('angular_search_window', 0.087)  # ~5 degrees in radians
        self.declare_parameter('depth_filter_threshold', 0.2)
        self.declare_parameter('target_frame', 'base_link')
        self.declare_parameter('publish_visualization', True)
        
        # Get parameters
        self.transform_timeout = self.get_parameter('transform_timeout').get_parameter_value().double_value
        self.sync_tolerance = self.get_parameter('sync_tolerance').get_parameter_value().double_value
        self.min_confidence = self.get_parameter('min_detection_confidence').get_parameter_value().double_value
        self.max_lidar_range = self.get_parameter('max_lidar_range').get_parameter_value().double_value
        self.min_lidar_range = self.get_parameter('min_lidar_range').get_parameter_value().double_value
        self.angular_window = self.get_parameter('angular_search_window').get_parameter_value().double_value
        self.depth_threshold = self.get_parameter('depth_filter_threshold').get_parameter_value().double_value
        self.target_frame = self.get_parameter('target_frame').get_parameter_value().string_value
        self.publish_viz = self.get_parameter('publish_visualization').get_parameter_value().bool_value
        
        # Camera calibration parameters 
        self.camera_matrix = np.array([
            [515.23999, 0.0, 327.98539],
            [0.0, 516.21085, 246.32969],
            [0.0, 0.0, 1.0]
        ])
        self.dist_coeffs = np.array([0.135681, -0.227115, 0.001488, 0.003070, 0.000000])
        self.image_width = 640
        self.image_height = 480
        
        # Extract camera intrinsics
        self.fx = self.camera_matrix[0, 0]
        self.fy = self.camera_matrix[1, 1]
        self.cx = self.camera_matrix[0, 2]
        self.cy = self.camera_matrix[1, 2]
        
        # TF2 setup
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        # QoS profiles
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            depth=10
        )
        qos_profile_detection = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            depth=10
        )
        
        # Pickup position publisher
        self.pickup_position_publisher = self.create_publisher(
            Point,
            '/pick_position',
            10
        )
        
        # Pickup detection subscriber with LiDAR sync
        self.pickup_detection_sub = message_filters.Subscriber(
            self,
            Detection2DArray,
            '/pickup_detection_bbox',
            qos_profile=qos_profile_detection
        )
        
        self.lidar_sub = message_filters.Subscriber(
            self,
            LaserScan,
            '/scan',
            qos_profile=qos_profile
        )
        
        # Synchronizer for pickup detections
        self.pickup_sync = message_filters.ApproximateTimeSynchronizer(
            [self.pickup_detection_sub, self.lidar_sub],
            queue_size=50,
            slop=self.sync_tolerance,
            allow_headerless=True
        )
        self.pickup_sync.registerCallback(self.pickup_fusion_callback)
        
        # Color mapping for visualization
        self.color_mapping = {
            'red': ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0),
            'blue': ColorRGBA(r=0.0, g=0.0, b=1.0, a=1.0),
            'white': ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0),
            'unknown': ColorRGBA(r=0.5, g=0.5, b=0.5, a=1.0)
        }
        
        self.get_logger().info('Sensor Fusion Node initialized')

        # Debug subscribers to check message flow
        self.debug_pickup_sub = self.create_subscription(
            Detection2DArray,
            '/pickup_detection_bbox',
            self.debug_pickup_callback,
            10
        )

        self.debug_lidar_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.debug_lidar_callback,
            10
        )

    def debug_pickup_callback(self, msg):
        """Debug callback for pickup detections"""
        self.get_logger().info(f'DEBUG: Pickup detection received with {len(msg.detections)} detections at time {msg.header.stamp.sec}.{msg.header.stamp.nanosec}')

    def debug_lidar_callback(self, msg):
        """Debug callback for LiDAR data"""
        # Only print every 10th message to avoid spam
        if hasattr(self, 'debug_lidar_counter'):
            self.debug_lidar_counter += 1
        else:
            self.debug_lidar_counter = 0
        
        if self.debug_lidar_counter % 10 == 0:
            self.get_logger().info(f'DEBUG: LiDAR scan received with {len(msg.ranges)} ranges at time {msg.header.stamp.sec}.{msg.header.stamp.nanosec}')

    def pickup_fusion_callback(self, pickup_detections_msg, lidar_msg):
        """Callback for pickup detection sensor fusion"""
        try:
            # DEBUG: Print LiDAR characteristics
            self.get_logger().info(f"=== LIDAR DEBUG INFO ===")
            self.get_logger().info(f"LiDAR ranges length: {len(lidar_msg.ranges)}")
            self.get_logger().info(f"LiDAR angle_min: {lidar_msg.angle_min:.3f} rad ({math.degrees(lidar_msg.angle_min):.1f} deg)")
            self.get_logger().info(f"LiDAR angle_max: {lidar_msg.angle_max:.3f} rad ({math.degrees(lidar_msg.angle_max):.1f} deg)")
            self.get_logger().info(f"LiDAR angle_increment: {lidar_msg.angle_increment:.6f} rad ({math.degrees(lidar_msg.angle_increment):.3f} deg)")
            self.get_logger().info(f"LiDAR range_min: {lidar_msg.range_min:.3f}")
            self.get_logger().info(f"LiDAR range_max: {lidar_msg.range_max:.3f}")
            
            # Check some actual range values
            valid_ranges = [r for r in lidar_msg.ranges if not math.isinf(r) and not math.isnan(r) and r > 0]
            if valid_ranges:
                self.get_logger().info(f"Valid ranges: min={min(valid_ranges):.3f}, max={max(valid_ranges):.3f}, count={len(valid_ranges)}")
            
            pickup_positions = []
            self.get_logger().info('pickup fusion callback')
            
            for detection in pickup_detections_msg.detections:
                self.get_logger().info(f'pixel coordinates: x={detection.bbox.center.position.x}, y={detection.bbox.center.position.y}')
                
                if len(detection.results) == 0:
                    continue
                    
                # Get detection confidence
                hypothesis = detection.results[0].hypothesis
                confidence = hypothesis.score
                
                # Filter by confidence
                if confidence < self.min_confidence:
                    continue
                
                # Get 2D detection center
                center_x = detection.bbox.center.position.x
                center_y = detection.bbox.center.position.y
                
                # Convert pixel coordinates to 3D pose
                pose_3d = self.pixel_to_3d_pose(
                    center_x, center_y, lidar_msg, pickup_detections_msg.header
                )
                
                if pose_3d is not None:
                    pickup_position = Point()
                    pickup_position.x = pose_3d.position.x - 0.03
                    pickup_position.y = pose_3d.position.y
                    pickup_position.z = pose_3d.position.z - 0.015
                    pickup_positions.append(pickup_position)
            
            # Publish pickup positions
            if pickup_positions:
                self.pickup_position_publisher.publish(pickup_positions[0])
                self.get_logger().info(f'Published pickup position: x={pickup_positions[0].x:.3f}, y={pickup_positions[0].y:.3f}, z={pickup_positions[0].z:.3f}')
                
        except Exception as e:
            self.get_logger().error(f'Error in pickup fusion callback: {e}')

    def pixel_to_3d_pose(self, pixel_x, pixel_y, lidar_msg, header):
        """Convert pixel coordinates to 3D pose using LiDAR depth"""
        try:
            # Step 1: Undistort pixel coordinates first
            pixel = np.array([[[pixel_x, pixel_y]]], dtype=np.float32)
            undistorted = cv2.undistortPoints(pixel, self.camera_matrix, self.dist_coeffs, P=self.camera_matrix)
            undist_x = undistorted[0, 0, 0]
            undist_y = undistorted[0, 0, 1]
            
            # Convert to normalized camera coordinates
            u = (undist_x - self.cx) / self.fx
            v = (undist_y - self.cy) / self.fy
            
            # Step 2: Create camera ray in optical frame
            # Camera optical frame: X-right, Y-down, Z-forward
            camera_ray = np.array([u, v, 1.0])
            camera_ray = camera_ray / np.linalg.norm(camera_ray)
            
            # Step 3: Transform ray to base_link frame
            try:
                transform = self.tf_buffer.lookup_transform(
                    self.target_frame,
                    'camera_rgb_optical_frame',
                    header.stamp,
                    timeout=rclpy.duration.Duration(seconds=self.transform_timeout)
                )
                
                # Transform the ray direction to base_link
                base_ray = self.transform_vector(camera_ray, transform)
                
                # Also get camera position in base_link
                camera_pos = np.array([
                    transform.transform.translation.x,
                    transform.transform.translation.y,
                    transform.transform.translation.z
                ])
                
            except TransformException as e:
                self.get_logger().warn(f'Transform lookup failed: {e}')
                return None
            
            # Step 4: Get multiple depth measurements around the target
            bearing_angle = math.atan2(base_ray[1], base_ray[0])
            
            # Get depth measurements with different strategies
            depth_measurement = self.get_lidar_depth(bearing_angle, lidar_msg)
            min_depth = self.get_minimum_depth_in_region(bearing_angle, lidar_msg)
            
            # Debug logging for LiDAR analysis
            self.get_logger().info(f'Bearing angle: {bearing_angle:.3f} rad ({math.degrees(bearing_angle):.1f} deg)')
            self.get_logger().info(f'Raw depth measurement: {depth_measurement}')
            self.get_logger().info(f'Minimum depth in region: {min_depth}')
            
            # Use the minimum depth if it's significantly different and more reasonable
            if min_depth is not None and depth_measurement is not None:
                if min_depth < depth_measurement * 0.7 and min_depth > 0.1:  # 30% threshold
                    self.get_logger().info(f'Using minimum depth {min_depth} instead of {depth_measurement}')
                    depth_measurement = min_depth
            
            if depth_measurement is None:
                return None
            
            # Step 5: Calculate 3D position using direct ray scaling
            # Since LiDAR gives us the actual distance to the object,
            # we scale the camera ray by this distance
            position_3d = camera_pos + base_ray * depth_measurement
            
            # Alternative calculation for comparison - using horizontal scaling
            horizontal_distance = depth_measurement
            ray_horizontal_norm = math.sqrt(base_ray[0]**2 + base_ray[1]**2)
            
            if ray_horizontal_norm > 1e-6:
                scale_factor = horizontal_distance / ray_horizontal_norm
                position_3d_alt = camera_pos + base_ray * scale_factor
                
                # Compare methods
                dist_diff = np.linalg.norm(position_3d - position_3d_alt)
                self.get_logger().info(f'Direct method: {position_3d}')
                self.get_logger().info(f'Horizontal scaling: {position_3d_alt}')
                self.get_logger().info(f'Difference: {dist_diff:.3f}')
                
                # Use horizontal scaling if the object is close (more accurate for nearby objects)
                if depth_measurement < 0.5:  # Less than 50cm
                    position_3d = position_3d_alt
                    self.get_logger().info('Using horizontal scaling for close object')
            
            # Create pose message
            pose = Pose()
            pose.position.x = float(position_3d[0] - 0.05) 
            pose.position.y = float(position_3d[1])
            pose.position.z = float(position_3d[2]+0.022)
            
            # Set orientation (identity for now)
            pose.orientation.x = 0.0
            pose.orientation.y = 0.0
            pose.orientation.z = 0.0
            pose.orientation.w = 1.0
            
            # Debug logging
            self.get_logger().info(f'Camera position: {camera_pos}')
            self.get_logger().info(f'Base ray direction: {base_ray}')
            self.get_logger().info(f'Final position: {position_3d}')
            
            return pose
            
        except Exception as e:
            self.get_logger().error(f'Error in pixel_to_3d_pose: {e}')
            return None

    def get_lidar_depth(self, bearing_angle, lidar_msg):
        """Get depth from LiDAR at specified bearing angle with improved filtering"""
        try:
            # Normalize angle to [angle_min, angle_max]
            angle_range = lidar_msg.angle_max - lidar_msg.angle_min
            
            # Handle angle wrapping
            normalized_angle = bearing_angle
            while normalized_angle < lidar_msg.angle_min:
                normalized_angle += 2 * math.pi
            while normalized_angle > lidar_msg.angle_max:
                normalized_angle -= 2 * math.pi
            
            # If still outside range, try the other direction
            if normalized_angle < lidar_msg.angle_min or normalized_angle > lidar_msg.angle_max:
                if bearing_angle > 0:
                    normalized_angle = bearing_angle - 2 * math.pi
                else:
                    normalized_angle = bearing_angle + 2 * math.pi
            
            # Calculate index
            relative_angle = normalized_angle - lidar_msg.angle_min
            index = int(round(relative_angle / lidar_msg.angle_increment))
            
            # Ensure index is within bounds
            if index < 0 or index >= len(lidar_msg.ranges):
                self.get_logger().warn(f'Index {index} out of bounds for angle {bearing_angle}')
                return None
            
            # Get range measurements in search window
            search_indices = self.get_search_indices(index, lidar_msg)
            valid_ranges = []
            
            for idx in search_indices:
                if 0 <= idx < len(lidar_msg.ranges):
                    range_val = lidar_msg.ranges[idx]
                    if (self.min_lidar_range <= range_val <= self.max_lidar_range and
                        not math.isinf(range_val) and not math.isnan(range_val)):
                        valid_ranges.append(range_val)
            
            if not valid_ranges:
                self.get_logger().warn(f'No valid ranges found for angle {bearing_angle}')
                return None
            
            # Use minimum distance for better object detection
            depth = min(valid_ranges)
            
            # Additional filtering - remove outliers
            if len(valid_ranges) > 3:
                sorted_ranges = sorted(valid_ranges)
                # Use the minimum of the first quartile
                quartile_index = max(1, len(sorted_ranges) // 4)
                depth = sorted_ranges[quartile_index - 1]
            
            return depth
            
        except Exception as e:
            self.get_logger().error(f'Error in get_lidar_depth: {e}')
            return None

    def get_search_indices(self, center_index, lidar_msg):
        """Get indices within angular search window"""
        # Calculate number of indices corresponding to search window
        window_indices = max(1, int(self.angular_window / lidar_msg.angle_increment))
        
        indices = []
        for i in range(-window_indices, window_indices + 1):
            idx = center_index + i
            indices.append(idx)
        
        return indices

    def get_minimum_depth_in_region(self, bearing_angle, lidar_msg):
        """Get minimum depth in a larger region around the bearing angle"""
        try:
            # Use a larger search window for finding the actual object
            larger_window = self.angular_window * 3  # 3x larger window
            
            # Normalize angle
            normalized_angle = bearing_angle
            while normalized_angle < lidar_msg.angle_min:
                normalized_angle += 2 * math.pi
            while normalized_angle > lidar_msg.angle_max:
                normalized_angle -= 2 * math.pi
            
            if normalized_angle < lidar_msg.angle_min or normalized_angle > lidar_msg.angle_max:
                if bearing_angle > 0:
                    normalized_angle = bearing_angle - 2 * math.pi
                else:
                    normalized_angle = bearing_angle + 2 * math.pi
            
            # Calculate center index
            relative_angle = normalized_angle - lidar_msg.angle_min
            center_index = int(round(relative_angle / lidar_msg.angle_increment))
            
            # Get search window
            window_indices = max(3, int(larger_window / lidar_msg.angle_increment))
            
            valid_ranges = []
            for i in range(-window_indices, window_indices + 1):
                idx = center_index + i
                if 0 <= idx < len(lidar_msg.ranges):
                    range_val = lidar_msg.ranges[idx]
                    if (self.min_lidar_range <= range_val <= self.max_lidar_range and
                        not math.isinf(range_val) and not math.isnan(range_val)):
                        valid_ranges.append(range_val)
            
            if not valid_ranges:
                return None
            
            # Return minimum distance (closest object)
            return min(valid_ranges)
            
        except Exception as e:
            self.get_logger().error(f'Error in get_minimum_depth_in_region: {e}')
            return None

    def transform_vector(self, vector, transform):
        """Transform a 3D vector using a transform"""
        # Extract rotation from transform
        q = transform.transform.rotation
        rotation = R.from_quat([q.x, q.y, q.z, q.w])
        
        # Apply rotation to vector (direction only, no translation)
        rotated_vector = rotation.apply(vector)
        
        return rotated_vector

def main(args=None):
    rclpy.init(args=args)
    node = SensorFusionNode()
    node.get_logger().info('Sensor Fusion Node is running')
    executor = MultiThreadedExecutor(num_threads=6)  
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
