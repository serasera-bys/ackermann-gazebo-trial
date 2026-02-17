import json
import math
from pathlib import Path
from typing import Optional

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String


class RLPolicyPlanner(Node):
    def __init__(self) -> None:
        super().__init__("hybrid_rl_planner")

        self.declare_parameter("rate_hz", 20.0)
        self.declare_parameter("goal_x", 8.0)
        self.declare_parameter("goal_y", 0.0)
        self.declare_parameter("reach_tolerance", 0.25)
        self.declare_parameter("max_linear_speed", 0.8)
        self.declare_parameter("max_angular_speed", 0.7)
        self.declare_parameter("output_topic", "/cmd_vel_raw")
        self.declare_parameter("odom_topic", "/ackermann_steering_controller/odometry")
        self.declare_parameter("scan_topic", "/scan")
        self.declare_parameter(
            "policy_file",
            "/home/bernard/ros2_ws/src/hybrid_nav_robot/experiments/rl_policy.json",
        )
        self.declare_parameter("default_k_linear", 0.8)
        self.declare_parameter("default_k_heading", 1.2)
        self.declare_parameter("default_k_avoid", 0.6)
        self.declare_parameter("avoid_distance", 1.1)
        self.declare_parameter("use_episode_events", True)
        self.declare_parameter("episode_event_topic", "/hybrid_nav/episode_event")

        self.rate_hz = float(self.get_parameter("rate_hz").value)
        self.goal_x = float(self.get_parameter("goal_x").value)
        self.goal_y = float(self.get_parameter("goal_y").value)
        self.reach_tolerance = float(self.get_parameter("reach_tolerance").value)
        self.max_linear_speed = float(self.get_parameter("max_linear_speed").value)
        self.max_angular_speed = float(self.get_parameter("max_angular_speed").value)
        self.output_topic = str(self.get_parameter("output_topic").value)
        self.odom_topic = str(self.get_parameter("odom_topic").value)
        self.scan_topic = str(self.get_parameter("scan_topic").value)
        self.policy_file = str(self.get_parameter("policy_file").value)
        self.k_linear = float(self.get_parameter("default_k_linear").value)
        self.k_heading = float(self.get_parameter("default_k_heading").value)
        self.k_avoid = float(self.get_parameter("default_k_avoid").value)
        self.avoid_distance = float(self.get_parameter("avoid_distance").value)
        self.use_episode_events = bool(self.get_parameter("use_episode_events").value)
        self.episode_event_topic = str(self.get_parameter("episode_event_topic").value)

        self.pose_x = 0.0
        self.pose_y = 0.0
        self.yaw = 0.0
        self.odom_received = False
        self.front_range = math.inf
        self.left_range = math.inf
        self.right_range = math.inf

        self._policy_mtime: Optional[float] = None
        self._load_policy(force=True)

        self.create_subscription(Odometry, self.odom_topic, self.on_odom, 10)
        self.create_subscription(LaserScan, self.scan_topic, self.on_scan, qos_profile_sensor_data)
        if self.use_episode_events:
            self.create_subscription(String, self.episode_event_topic, self.on_episode_event, 20)
        self.cmd_pub = self.create_publisher(Twist, self.output_topic, 10)
        period = 1.0 / max(self.rate_hz, 1.0)
        self.create_timer(period, self.on_tick)
        self.create_timer(1.0, self._load_policy)

        self.get_logger().info(
            f"RL policy planner started: goal=({self.goal_x:.2f}, {self.goal_y:.2f}), "
            f"policy_file={self.policy_file}, output={self.output_topic}"
        )

    def on_episode_event(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        if str(payload.get("event", "")).strip() != "episode_start":
            return
        self.goal_x = float(payload.get("goal_x", self.goal_x))
        self.goal_y = float(payload.get("goal_y", self.goal_y))

    def _load_policy(self, force: bool = False) -> None:
        path = Path(self.policy_file)
        if not path.exists():
            return
        mtime = path.stat().st_mtime
        if not force and self._policy_mtime is not None and mtime <= self._policy_mtime:
            return

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            self.get_logger().warn(f"Failed to load policy file {path}: {exc}")
            return

        self.k_linear = float(data.get("k_linear", self.k_linear))
        self.k_heading = float(data.get("k_heading", self.k_heading))
        self.k_avoid = float(data.get("k_avoid", self.k_avoid))
        self.avoid_distance = float(data.get("avoid_distance", self.avoid_distance))
        self._policy_mtime = mtime
        self.get_logger().info(
            f"Loaded policy: k_linear={self.k_linear:.3f}, "
            f"k_heading={self.k_heading:.3f}, k_avoid={self.k_avoid:.3f}, "
            f"avoid_distance={self.avoid_distance:.2f}"
        )

    def on_odom(self, msg: Odometry) -> None:
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        self.pose_x = p.x
        self.pose_y = p.y
        self.yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z),
        )
        self.odom_received = True

    def on_scan(self, msg: LaserScan) -> None:
        front_min = math.inf
        left_min = math.inf
        right_min = math.inf
        for i, r in enumerate(msg.ranges):
            if not math.isfinite(r):
                continue
            if r < msg.range_min or r > msg.range_max:
                continue
            angle = msg.angle_min + i * msg.angle_increment
            abs_angle = abs(angle)
            if abs_angle <= 0.35:
                front_min = min(front_min, r)
            elif 0.35 < angle <= 1.30:
                left_min = min(left_min, r)
            elif -1.30 <= angle < -0.35:
                right_min = min(right_min, r)

        self.front_range = front_min
        self.left_range = left_min
        self.right_range = right_min

    def on_tick(self) -> None:
        cmd = Twist()
        if not self.odom_received:
            self.cmd_pub.publish(cmd)
            return

        dx = self.goal_x - self.pose_x
        dy = self.goal_y - self.pose_y
        dist = math.hypot(dx, dy)
        if dist < self.reach_tolerance:
            self.cmd_pub.publish(cmd)
            return

        heading_target = math.atan2(dy, dx)
        heading_error = self._wrap_to_pi(heading_target - self.yaw)
        avoid_feature = self._compute_avoid_feature()

        linear = self.k_linear * dist * self._front_speed_scale()
        angular = self.k_heading * heading_error + self.k_avoid * avoid_feature

        cmd.linear.x = self._clamp(linear, 0.0, self.max_linear_speed)
        cmd.angular.z = self._clamp(angular, -self.max_angular_speed, self.max_angular_speed)
        self.cmd_pub.publish(cmd)

    def _front_speed_scale(self) -> float:
        if not math.isfinite(self.front_range):
            return 1.0
        ratio = self.front_range / max(self.avoid_distance, 1e-3)
        return self._clamp(ratio, 0.2, 1.0)

    def _compute_avoid_feature(self) -> float:
        left = self.left_range if math.isfinite(self.left_range) else self.avoid_distance
        right = self.right_range if math.isfinite(self.right_range) else self.avoid_distance
        diff = left - right
        return self._clamp(diff, -1.0, 1.0)

    @staticmethod
    def _wrap_to_pi(angle: float) -> float:
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle

    @staticmethod
    def _clamp(value: float, min_v: float, max_v: float) -> float:
        return max(min_v, min(max_v, value))


def main() -> None:
    rclpy.init()
    node = RLPolicyPlanner()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
