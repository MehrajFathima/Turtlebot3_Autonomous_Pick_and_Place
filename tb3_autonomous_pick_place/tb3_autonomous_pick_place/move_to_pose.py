#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from std_msgs.msg import Float64MultiArray, String
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import (
    Constraints,
    PlanningOptions,
    MotionPlanRequest,
    JointConstraint
)
from rclpy.action import ActionClient
from control_msgs.action import GripperCommand

class SimpleMoveItNode(Node):
    def __init__(self):
        super().__init__('simple_moveit_node')
        self._ready_timer = None
        # MoveIt and gripper action clients
        self.moveit_client = ActionClient(self, MoveGroup, '/move_action')
        self.gripper_client = ActionClient(self, GripperCommand, '/gripper_controller/gripper_cmd')

        self.planning_group = "arm"
        self.joint_names = ["joint1", "joint2", "joint3", "joint4"]

        # Publishers for pick and place status
        self.pick_status_pub = self.create_publisher(String, '/pick_status', 10)
        self.place_status_pub = self.create_publisher(String, '/place_status', 10)

        # Wait for action servers
        self.get_logger().info('Waiting for MoveGroup action server...')
        self.moveit_client.wait_for_server()
        self.get_logger().info('Waiting for Gripper action server...')
        self.gripper_client.wait_for_server()
        self.get_logger().info('SimpleMoveItNode initialized and ready.')

        # Subscribers for pick and place
        self.create_subscription(Float64MultiArray, '/pick_joint_positions', self.pick_callback, 10)
        self.create_subscription(Float64MultiArray, '/place_joint_positions', self.place_callback, 10)

    def _go_to_ready_once(self):
            self.get_logger().info('Returning to ready position...')
            self.go_to_ready_position()
            if self._ready_timer:
                self._ready_timer.cancel()
                self._ready_timer = None

    # ----------- JOINT GOAL SENDER (ASYNC) -----------
    def send_joint_goal(self, joint_positions, done_cb=None):
        goal_msg = MoveGroup.Goal()
        goal_msg.request = MotionPlanRequest()
        goal_msg.request.group_name = self.planning_group
        goal_msg.request.num_planning_attempts = 10
        goal_msg.request.allowed_planning_time = 5.0
        goal_msg.request.planner_id = ""
        constraints = Constraints()
        for joint_name, position in zip(self.joint_names, joint_positions):
            jc = JointConstraint()
            jc.joint_name = joint_name
            jc.position = position
            jc.tolerance_above = 0.01
            jc.tolerance_below = 0.01
            jc.weight = 1.0
            constraints.joint_constraints.append(jc)
        goal_msg.request.goal_constraints = [constraints]
        goal_msg.planning_options = PlanningOptions()
        goal_msg.planning_options.plan_only = False
        goal_msg.planning_options.replan = True
        goal_msg.planning_options.replan_attempts = 5
        self.get_logger().info(f'Sending joint goal: {joint_positions}')

        send_goal_future = self.moveit_client.send_goal_async(
            goal_msg,
            feedback_callback=self.feedback_callback
        )
        def goal_response_callback(fut):
            goal_handle = fut.result()
            if not goal_handle.accepted:
                self.get_logger().error('Goal rejected!')
                if done_cb:
                    done_cb(False)
                return
            result_future = goal_handle.get_result_async()
            def result_callback(fut2):
                result = fut2.result().result
                if result.error_code.val == 1:
                    self.get_logger().info('Goal completed successfully!')
                    if done_cb:
                        done_cb(True)
                else:
                    self.get_logger().error(f'Goal failed with error code: {result.error_code.val}')
                    if done_cb:
                        done_cb(False)
            result_future.add_done_callback(result_callback)
        send_goal_future.add_done_callback(goal_response_callback)

    def feedback_callback(self, feedback_msg):
        self.get_logger().info(f'Feedback: {feedback_msg.feedback.state}')

    # ----------- GRIPPER COMMANDS (ASYNC) -----------
    def close_gripper(self, done_cb=None):
        goal_msg = GripperCommand.Goal()
        goal_msg.command.position = -0.01
        goal_msg.command.max_effort = 0.0
        self.get_logger().info('Closing gripper...')
        send_goal_future = self.gripper_client.send_goal_async(goal_msg)
        def goal_response_callback(fut):
            goal_handle = fut.result()
            if not goal_handle.accepted:
                self.get_logger().error('Gripper goal was rejected.')
                if done_cb:
                    done_cb(False)
                return
            result_future = goal_handle.get_result_async()
            def result_callback(fut2):
                self.get_logger().info('Gripper closed.')
                if done_cb:
                    done_cb(True)
            result_future.add_done_callback(result_callback)
        send_goal_future.add_done_callback(goal_response_callback)

    def open_gripper(self, done_cb=None):
        goal_msg = GripperCommand.Goal()
        goal_msg.command.position = 0.01
        goal_msg.command.max_effort = 0.0
        self.get_logger().info('Opening gripper...')
        send_goal_future = self.gripper_client.send_goal_async(goal_msg)
        def goal_response_callback(fut):
            goal_handle = fut.result()
            if not goal_handle.accepted:
                self.get_logger().error('Gripper goal was rejected.')
                if done_cb:
                    done_cb(False)
                return
            result_future = goal_handle.get_result_async()
            def result_callback(fut2):
                self.get_logger().info('Gripper opened.')
                if done_cb:
                    done_cb(True)
            result_future.add_done_callback(result_callback)
        send_goal_future.add_done_callback(goal_response_callback)

    # ----------- READY POSITION (ASYNC) -----------
    def go_to_ready_position(self, done_cb=None):
        ready_joints = [0.0, -1.0, 0.3, 0.7]
        self.send_joint_goal(ready_joints, done_cb=done_cb)

    # ----------- PICK AND PLACE CALLBACKS -----------
    def pick_callback(self, msg):
        self.get_logger().info('pick_callback triggered')
        joint_positions = msg.data
        self.open_gripper()
        
        def after_joint_goal(success):
            if success:
                self.get_logger().info('Pick position reached, closing gripper...')
                self.close_gripper(done_cb=after_gripper)
            else:
                self.get_logger().error('Failed to reach pick position')
        def after_gripper(success):
            if success:
                self.get_logger().info('Returning to ready position...')
                # Publish pick success message
                pick_msg = String()
                pick_msg.data = "object_picked"
                self.pick_status_pub.publish(pick_msg)
                self._ready_timer = self.create_timer(0.50, self._go_to_ready_once)
                #self.go_to_ready_position()
            else:
                self.get_logger().error('Failed to close gripper')
        self.send_joint_goal(joint_positions, done_cb=after_joint_goal)

    def place_callback(self, msg):
        self.get_logger().info('place_callback triggered')
        joint_positions = msg.data
        def after_joint_goal(success):
            if success:
                self.get_logger().info('Place position reached, opening gripper...')
                self.open_gripper(done_cb=after_gripper)
            else:
                self.get_logger().error('Failed to reach place position')
        def after_gripper(success):
            if success:
                self.get_logger().info('Returning to ready position...')
                # Publish place success message
                place_msg = String()
                place_msg.data = "object_placed"
                self.place_status_pub.publish(place_msg)
                self._ready_timer = self.create_timer(0.50, self._go_to_ready_once)
                #self.go_to_ready_position()
            else:
                self.get_logger().error('Failed to open gripper')
        self.send_joint_goal(joint_positions, done_cb=after_joint_goal)

def main(args=None):
    rclpy.init(args=args)
    node = SimpleMoveItNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        node.get_logger().info('Interrupted by user')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()