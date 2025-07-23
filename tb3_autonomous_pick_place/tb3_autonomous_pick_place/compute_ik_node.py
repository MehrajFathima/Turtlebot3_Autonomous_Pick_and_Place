#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

from moveit_msgs.srv import GetPositionIK
from geometry_msgs.msg import Pose, Point, Quaternion, PoseStamped, Twist
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray, String

import threading
import math

class TurtleBot3ArmIK(Node):
    def __init__(self):
        super().__init__('tb3_arm_ik_controller')
        self.callback_group = ReentrantCallbackGroup()
        self.arm_group_name = "arm"
        self.planning_frame = "base_link"
        self.end_effector_link = "end_effector_link"
        self.arm_joint_names = [
            'joint1',
            'joint2', 
            'joint3',
            'joint4'
        ]
        self.current_joint_states = None
        self.joint_states_lock = threading.Lock()

        # Store last target positions for retry logic
        self.last_target_x = None
        self.last_target_y = None
        self.last_target_z = None
        self.last_which = None

        # IK service client
        self.ik_service = self.create_client(
            GetPositionIK,
            '/compute_ik',
            callback_group=self.callback_group
        )

        # Subscribers
        self.joint_state_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10,
            callback_group=self.callback_group
        )
        self.pick_position_sub = self.create_subscription(
            Point,
            '/pick_position',
            self.pick_position_callback,
            10,
            callback_group=self.callback_group
        )
        self.place_position_sub = self.create_subscription(
            Point,
            '/place_position',
            self.place_position_callback,
            10,
            callback_group=self.callback_group
        )

        self.place_object_sub = self.create_subscription(
            String,
            '/place_object',
            self.place_object_callback,
            10,
            callback_group=self.callback_group
        )

        # Publishers for joint values
        self.pick_joint_pub = self.create_publisher(
            Float64MultiArray,
            '/pick_joint_positions',
            10
        )
        self.place_joint_pub = self.create_publisher(
            Float64MultiArray,
            '/place_joint_positions',
            10
        )

        # Publisher for robot movement (cmd_vel)
        self.cmd_vel_pub = self.create_publisher(
            Twist,
            '/cmd_vel',
            10
        )

        # Publisher for target color
        self.target_color_pub = self.create_publisher(
            String,
            '/target_color',
            10
        )

        self.planning_time = 5.0

        self.get_logger().info("TurtleBot3 OpenManipulator IK Controller Initialized")
        self.get_logger().info(f"Planning frame: {self.planning_frame}")
        self.get_logger().info(f"End effector link: {self.end_effector_link}")

        self.wait_for_services()

    def wait_for_services(self):
        self.get_logger().info("Waiting for IK service...")
        if not self.ik_service.wait_for_service(timeout_sec=10.0):
            self.get_logger().warn("IK service not available")
        else:
            self.get_logger().info("IK service connected")

    def joint_state_callback(self, msg):
        with self.joint_states_lock:
            self.current_joint_states = msg

    def get_current_joint_values(self):
        with self.joint_states_lock:
            if self.current_joint_states is None:
                return None
            joint_values = []
            for joint_name in self.arm_joint_names:
                try:
                    idx = self.current_joint_states.name.index(joint_name)
                    joint_values.append(self.current_joint_states.position[idx])
                except ValueError:
                    self.get_logger().warn(f"Joint {joint_name} not found in joint states")
                    return None
            return joint_values

    def pick_position_callback(self, msg):
        x, y, z = msg.x, msg.y, msg.z
        self.get_logger().info(f"Received pick position: x={x:.3f}, y={y:.3f}, z={z:.3f}")
        self.compute_ik_async(x, y, z, which='pick')

    def place_position_callback(self, msg):
        x, y, z = msg.x, msg.y, msg.z
        self.get_logger().info(f"Received place position: x={x:.3f}, y={y:.3f}, z={z:.3f}")
        self.compute_ik_async(x, y, z, which='place')

    def place_object_callback(self, msg):
        self.compute_ik_async(0.28, 0.0106, 0.199, which='place')


    def move_robot_closer(self):
        """Move the robot 1 cm closer to the target"""
        if self.last_target_x is None or self.last_target_y is None:
            self.get_logger().warn("No previous target position to move closer to")
            return

        # Calculate direction to target
        distance_to_target = math.sqrt(self.last_target_x**2 + self.last_target_y**2)
        
        if distance_to_target > 0:
            # Unit vector towards target
            unit_x = self.last_target_x / distance_to_target
            unit_y = self.last_target_y / distance_to_target
            
            # Move 1 cm (0.01 m) towards target
            move_distance = 0.05
            
            # Create twist message for forward movement
            twist = Twist()
            twist.linear.x = move_distance  # Move forward 1 cm
            twist.linear.y = 0.0
            twist.linear.z = 0.0
            twist.angular.x = 0.0
            twist.angular.y = 0.0
            twist.angular.z = 0.0
            
            # Publish movement command
            self.cmd_vel_pub.publish(twist)
            self.get_logger().info(f"Moving robot 1 cm closer to target")
            
            # Stop the robot after a brief moment (you might want to adjust this)
            self.create_timer(3.0, self.stop_robot)

    def stop_robot(self):
        """Stop the robot movement"""
        twist = Twist()
        self.cmd_vel_pub.publish(twist)

    def publish_target_color(self):
        """Publish target color as 'red'"""
        color_msg = String()
        color_msg.data = 'blue'
        self.target_color_pub.publish(color_msg)
        self.get_logger().info("Published target color: red")

    def compute_ik_async(self, x, y, z, orientation=None, which='pick'):
        # Store target position for potential retry
        self.last_target_x = x
        self.last_target_y = y
        self.last_target_z = z
        self.last_which = which

        pose_target = Pose()
        pose_target.position = Point(x=x, y=y, z=z)
        if orientation is None:
            pose_target.orientation = Quaternion(x=0.0, y=0.707, z=0.0, w=0.707)
        else:
            pose_target.orientation = orientation
        self.get_logger().info(f"Computing IK for position: x={x:.3f}, y={y:.3f}, z={z:.3f}")

        current_joint_values = self.get_current_joint_values()
        if current_joint_values is None:
            self.get_logger().warn("No current joint values available yet.")
            return

        ik_request = GetPositionIK.Request()
        ik_request.ik_request.group_name = self.arm_group_name
        ik_request.ik_request.robot_state.joint_state.header.frame_id = self.planning_frame
        ik_request.ik_request.robot_state.joint_state.header.stamp = self.get_clock().now().to_msg()
        ik_request.ik_request.robot_state.joint_state.name = self.arm_joint_names
        ik_request.ik_request.robot_state.joint_state.position = current_joint_values
        pose_stamped = PoseStamped()
        pose_stamped.header.frame_id = self.planning_frame
        pose_stamped.header.stamp = self.get_clock().now().to_msg()
        pose_stamped.pose = pose_target
        ik_request.ik_request.pose_stamped = pose_stamped
        ik_request.ik_request.ik_link_name = self.end_effector_link
        ik_request.ik_request.timeout.sec = int(self.planning_time)
        ik_request.ik_request.timeout.nanosec = int((self.planning_time % 1) * 1e9)

        # Pass which ('pick' or 'place') to the callback using a lambda
        future = self.ik_service.call_async(ik_request)
        future.add_done_callback(lambda fut: self.ik_result_callback(fut, which))

    def ik_result_callback(self, future, which):
        try:
            response = future.result()
            if response.error_code.val == response.error_code.SUCCESS:
                joint_values = []
                for joint_name in self.arm_joint_names:
                    try:
                        idx = response.solution.joint_state.name.index(joint_name)
                        joint_values.append(response.solution.joint_state.position[idx])
                    except ValueError:
                        self.get_logger().warn(f"Joint {joint_name} not found in IK solution")
                        return
                self.get_logger().info(f"{which.capitalize()} joint values: {joint_values}")
                msg = Float64MultiArray()
                msg.data = joint_values
                if which == 'pick':
                    self.pick_joint_pub.publish(msg)
                    print("Published pick joint values:", joint_values)
                elif which == 'place':
                    self.place_joint_pub.publish(msg)
                    print("Published place joint values:", joint_values)
            else:
                self.get_logger().error(f"IK failed with error code: {response.error_code.val}")
                # Move robot 1 cm closer and publish target color
                self.move_robot_closer()
                self.publish_target_color()
        except Exception as e:
            self.get_logger().error(f"IK service call exception: {e}")

def main():
    rclpy.init()
    try:
        arm_controller = TurtleBot3ArmIK()
        executor = MultiThreadedExecutor()
        executor.add_node(arm_controller)
        executor.spin()
    except Exception as e:
        print(f"Error: {e}")
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main()
