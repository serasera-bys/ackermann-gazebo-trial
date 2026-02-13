import json
import math
from pathlib import Path
from typing import Optional

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool


class MetricsLoggerNode(Node):
    def __init__(self) -> None:
        super().__init__("hybrid_metrics_logger")

        self.declare_parameter(
            "output_file",
            "/home/bernard/ros2_ws/src/hybrid_nav_robot/experiments/latest_metrics.json",
        )
        self.declare_parameter("odom_topic", "/odom")
        self.declare_parameter("cmd_topic", "/cmd_vel")
        self.declare_parameter("scan_topic", "/scan")
        self.declare_parameter("safety_topic", "/safety_layer/intervention")
        self.declare_parameter("goal_x", 8.0)
        self.declare_parameter("goal_y", 0.0)
        self.declare_parameter("goal_tolerance", 0.4)
        self.declare_parameter("collision_distance", 0.2)

        self.output_file = str(self.get_parameter("output_file").value)
        odom_topic = str(self.get_parameter("odom_topic").value)
        cmd_topic = str(self.get_parameter("cmd_topic").value)
        scan_topic = str(self.get_parameter("scan_topic").value)
        safety_topic = str(self.get_parameter("safety_topic").value)
        self.goal_x = float(self.get_parameter("goal_x").value)
        self.goal_y = float(self.get_parameter("goal_y").value)
        self.goal_tolerance = float(self.get_parameter("goal_tolerance").value)
        self.collision_distance = float(self.get_parameter("collision_distance").value)

        self.start_time = self.get_clock().now()
        self.last_x: Optional[float] = None
        self.last_y: Optional[float] = None
        self.distance_traveled = 0.0
        self.max_linear_cmd = 0.0
        self.max_angular_cmd = 0.0
        self.min_obstacle_range = math.inf
        self.safety_interventions = 0
        self.odom_samples = 0
        self.goal_distance = math.inf
        self.goal_reached = False
        self.time_to_goal_sec: Optional[float] = None
        self.collision_flag = False

        self.create_subscription(Odometry, odom_topic, self.on_odom, 20)
        self.create_subscription(Twist, cmd_topic, self.on_cmd, 20)
        self.create_subscription(LaserScan, scan_topic, self.on_scan, 10)
        self.create_subscription(Bool, safety_topic, self.on_safety, 10)
        self.create_timer(2.0, self.on_heartbeat)

        rclpy.get_default_context().on_shutdown(self.on_shutdown)
        self.get_logger().info(f"Metrics logger started, output={self.output_file}")

    def on_odom(self, msg: Odometry) -> None:
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        if self.last_x is not None and self.last_y is not None:
            self.distance_traveled += math.hypot(x - self.last_x, y - self.last_y)
        self.last_x = x
        self.last_y = y
        self.odom_samples += 1
        self.goal_distance = math.hypot(self.goal_x - x, self.goal_y - y)
        if not self.goal_reached and self.goal_distance <= self.goal_tolerance:
            self.goal_reached = True
            self.time_to_goal_sec = (
                (self.get_clock().now() - self.start_time).nanoseconds / 1e9
            )

    def on_cmd(self, msg: Twist) -> None:
        self.max_linear_cmd = max(self.max_linear_cmd, abs(msg.linear.x))
        self.max_angular_cmd = max(self.max_angular_cmd, abs(msg.angular.z))

    def on_scan(self, msg: LaserScan) -> None:
        finite_ranges = [r for r in msg.ranges if math.isfinite(r) and r >= msg.range_min]
        if finite_ranges:
            self.min_obstacle_range = min(self.min_obstacle_range, min(finite_ranges))
            if self.min_obstacle_range <= self.collision_distance:
                self.collision_flag = True

    def on_safety(self, msg: Bool) -> None:
        if msg.data:
            self.safety_interventions += 1

    def on_heartbeat(self) -> None:
        runtime = (self.get_clock().now() - self.start_time).nanoseconds / 1e9
        self.get_logger().info(
            f"runtime={runtime:.1f}s, dist={self.distance_traveled:.2f}m, "
            f"goal_dist={self._safe_range(self.goal_distance):.2f}m, "
            f"min_range={self._safe_range(self.min_obstacle_range):.2f}m, "
            f"safety_hits={self.safety_interventions}"
        )

    def on_shutdown(self) -> None:
        runtime = (self.get_clock().now() - self.start_time).nanoseconds / 1e9
        data = {
            "runtime_sec": runtime,
            "distance_traveled_m": self.distance_traveled,
            "max_linear_cmd_mps": self.max_linear_cmd,
            "max_angular_cmd_rps": self.max_angular_cmd,
            "min_obstacle_range_m": self._safe_range(self.min_obstacle_range),
            "safety_interventions": self.safety_interventions,
            "odom_samples": self.odom_samples,
            "goal_x": self.goal_x,
            "goal_y": self.goal_y,
            "goal_tolerance_m": self.goal_tolerance,
            "goal_distance_m": self._safe_range(self.goal_distance),
            "goal_reached": self.goal_reached,
            "time_to_goal_sec": -1.0 if self.time_to_goal_sec is None else self.time_to_goal_sec,
            "collision_flag": self.collision_flag,
        }

        output = Path(self.output_file)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(data, indent=2), encoding="utf-8")
        self.get_logger().info(f"Saved metrics to {output}")

    @staticmethod
    def _safe_range(value: float) -> float:
        return value if math.isfinite(value) else -1.0


def main() -> None:
    rclpy.init()
    node = MetricsLoggerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
