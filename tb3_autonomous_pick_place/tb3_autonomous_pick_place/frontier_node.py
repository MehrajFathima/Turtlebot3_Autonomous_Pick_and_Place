import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseStamped, Point
from visualization_msgs.msg import Marker
from nav2_msgs.action import NavigateToPose
from nav2_msgs.srv import SaveMap
from rclpy.action import ActionClient
from rclpy.qos import QoSProfile
from std_msgs.msg import Bool
from ament_index_python.packages import get_package_share_directory
from tf2_ros import Buffer, TransformListener, TransformException
import math
import os


class FrontierExplorer(Node):
    def __init__(self):
        super().__init__('frontier_explorer')

        self.FREE_SPACE_VALUES = {0, 42, 44, 45, 46, 47, 48, 49}
        self.UNKNOWN_CELL = -1
        self.HOME_REACHED_TOLERANCE = 0.3  # ✅ Added for cleaner config

        self.current_goal = None
        self.failed_frontiers = set()
        self.initial_position = None
        self.returning_home = False
        self.exploration_done = False

        qos = QoSProfile(depth=10)
        self.map_sub = self.create_subscription(OccupancyGrid, '/map', self.map_callback, qos)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.nav_to_pose_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self.map_saver_cli = self.create_client(SaveMap, '/map_saver/save_map')

        self.get_logger().info("Waiting for navigation action server...")
        self.nav_to_pose_client.wait_for_server()
        self.get_logger().info("Navigation action server available")

        self.marker_pub = self.create_publisher(Marker, 'frontier_markers', 10)
        self.completion_pub = self.create_publisher(Bool, '/exploration_complete', 10)

        self.latest_map = None
        self.timer = self.create_timer(30.0, self.explore)
        self.get_logger().info("Frontier explorer node started.")

        self.map_folder = os.path.join(get_package_share_directory('tb3_autonomous_pick_place'), 'generated_maps')
        os.makedirs(self.map_folder, exist_ok=True)

    def map_callback(self, msg: OccupancyGrid):
        self.latest_map = msg
        if self.initial_position is None:
            pose = self.get_robot_pose()
            if pose:
                self.initial_position = (pose.x, pose.y)
                self.get_logger().info(f"Initial position recorded: {self.initial_position}")

    def get_robot_pose(self):
        try:
            trans = self.tf_buffer.lookup_transform(
                'map',
                'base_link',
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=1.0)
            )
            return trans.transform.translation
        except TransformException as ex:
            self.get_logger().warn(f"TF lookup failed: {ex}")
            return None

    def explore(self):
        if self.latest_map is None or self.returning_home or self.exploration_done:
            return

        if self.current_goal is not None:
            self.get_logger().debug("A goal is already in progress. Skipping explore cycle.")
            return

        robot_pose = self.get_robot_pose()
        if robot_pose is None:
            return

        frontiers = self.detect_frontiers(self.latest_map)
        self.publish_frontiers_markers(frontiers)

        valid_frontiers = []
        for f in frontiers:
            if f in self.failed_frontiers:
                continue
            if not self.is_reachable(f):
                self.get_logger().warn(f"Frontier at {f} is not reachable.")
                self.failed_frontiers.add(f)
                continue
            valid_frontiers.append(f)

        if not valid_frontiers:
            self.get_logger().info("All frontiers explored or unreachable. Saving map and returning to start...")
            self.timer.cancel()
            self.save_map()
            return

        farthest_frontier = max(
            valid_frontiers,
            key=lambda f: self.euclidean_distance((robot_pose.x, robot_pose.y), f)
        )

        self.send_goal(farthest_frontier)

    def is_reachable(self, frontier, threshold=0.3):
        if self.latest_map is None:
            return False

        map_data = self.latest_map.data
        width = self.latest_map.info.width
        height = self.latest_map.info.height
        resolution = self.latest_map.info.resolution
        origin_x = self.latest_map.info.origin.position.x
        origin_y = self.latest_map.info.origin.position.y

        x, y = frontier
        grid_x = int((x - origin_x) / resolution)
        grid_y = int((y - origin_y) / resolution)

        def index(x, y):
            return y * width + x

        radius_cells = int(threshold / resolution)

        for dy in range(-radius_cells, radius_cells + 1):
            for dx in range(-radius_cells, radius_cells + 1):
                nx = grid_x + dx
                ny = grid_y + dy
                if 0 <= nx < width and 0 <= ny < height:
                    idx = index(nx, ny)
                    if map_data[idx] > 50:
                        return False
        return True

    def detect_frontiers(self, map_msg):
        width = map_msg.info.width
        height = map_msg.info.height
        data = map_msg.data

        def index(x, y):
            return y * width + x

        frontiers = []
        for y in range(1, height - 1):
            for x in range(1, width - 1):
                idx = index(x, y)
                if data[idx] not in self.FREE_SPACE_VALUES:
                    continue

                neighbors = [
                    data[index(x + 1, y)],
                    data[index(x - 1, y)],
                    data[index(x, y + 1)],
                    data[index(x, y - 1)]
                ]
                if self.UNKNOWN_CELL in neighbors:
                    map_x = map_msg.info.origin.position.x + (x + 0.5) * map_msg.info.resolution
                    map_y = map_msg.info.origin.position.y + (y + 0.5) * map_msg.info.resolution
                    frontiers.append((map_x, map_y))

        self.get_logger().info(f"Detected {len(frontiers)} frontier cells.")
        return frontiers

    def publish_frontiers_markers(self, frontiers):
        marker = Marker()
        marker.header.frame_id = self.latest_map.header.frame_id
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "frontiers"
        marker.id = 0
        marker.type = Marker.POINTS
        marker.action = Marker.ADD if frontiers else Marker.DELETE
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.1
        marker.scale.y = 0.1
        marker.color.a = 1.0
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0

        if frontiers:
            for fx, fy in frontiers:
                p = Point()
                p.x = fx
                p.y = fy
                p.z = 0.0
                marker.points.append(p)

        self.marker_pub.publish(marker)

    def send_goal(self, target):
        if self.current_goal is not None:
            self.get_logger().warn("Attempted to send a new goal while one is already active.")
            return

        goal_pose = PoseStamped()
        goal_pose.header.frame_id = self.latest_map.header.frame_id
        goal_pose.header.stamp = self.get_clock().now().to_msg()
        goal_pose.pose.position.x = target[0]
        goal_pose.pose.position.y = target[1]
        goal_pose.pose.position.z = 0.0
        goal_pose.pose.orientation.w = 1.0

        self.get_logger().info(f"Sending goal to: ({target[0]:.2f}, {target[1]:.2f})")

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = goal_pose

        self.current_goal = target
        future = self.nav_to_pose_client.send_goal_async(goal_msg)
        future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warn("Goal rejected by navigation server")
            self.failed_frontiers.add(self.current_goal)
            self.current_goal = None
            return

        self.get_logger().info("Goal accepted by navigation server")
        self._result_future = goal_handle.get_result_async()
        self._result_future.add_done_callback(self.get_result_callback)

    def get_result_callback(self, future):
        result = future.result().result
        status = future.result().status

        # ✅ Special case for returning home
        if self.returning_home:
            robot_pose = self.get_robot_pose()
            if robot_pose:
                distance_to_home = self.euclidean_distance(
                    (robot_pose.x, robot_pose.y), self.initial_position
                )
                if distance_to_home < self.HOME_REACHED_TOLERANCE:
                    self.get_logger().info("Already at initial position. Exploration complete.")
                    self.returning_home = False
                    self.exploration_done = True
                    self.completion_pub.publish(Bool(data=True))
                    self.get_logger().info("✅ Exploration complete flag published to /exploration_complete.")
                    self.current_goal = None
                    return

        if status == 4:
            self.get_logger().warn("Goal aborted.")
            self.failed_frontiers.add(self.current_goal)
        elif status == 5:
            self.get_logger().warn("Goal rejected.")
            self.failed_frontiers.add(self.current_goal)
        else:
            self.get_logger().info("Goal succeeded or completed.")
            if self.returning_home:
                self.returning_home = False
                self.exploration_done = True
                self.get_logger().info("Returned to initial position. Exploration complete.")
                self.completion_pub.publish(Bool(data=True))
                self.get_logger().info("✅ Exploration complete flag published to /exploration_complete.")

        self.current_goal = None
        if not self.exploration_done:
            self.explore()

    def navigate_to_initial_position(self):
        if self.initial_position is None:
            self.get_logger().warn("Initial position unknown. Cannot return home.")
            return

        self.returning_home = True
        self.send_goal(self.initial_position)

    def save_map(self):
        if not self.map_saver_cli.wait_for_service(timeout_sec=5.0):
            self.get_logger().warn("Map saver service not available")
            self.navigate_to_initial_position()
            return

        timestamp = self.get_clock().now().to_msg().sec
        map_name = f"explored_map_{timestamp}"
        full_path = os.path.join(self.map_folder, map_name)

        request = SaveMap.Request()
        request.map_topic = '/map'
        request.map_url = full_path
        request.image_format = 'png'

        self.get_logger().info(f"Saving map to {full_path}.pgm/.yaml")

        def on_map_saved(fut):
            try:
                result = fut.result()
                if result.success:
                    self.get_logger().info("Map saved successfully.")
                else:
                    self.get_logger().warn("Map save failed.")
            except Exception as e:
                self.get_logger().error(f"Map saving service call failed: {e}")
            finally:
                self.navigate_to_initial_position()

        future = self.map_saver_cli.call_async(request)
        future.add_done_callback(on_map_saved)

    @staticmethod
    def euclidean_distance(p1, p2):
        return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def main(args=None):
    rclpy.init(args=args)
    node = FrontierExplorer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

