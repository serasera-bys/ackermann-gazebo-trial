import json
import math

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String


class RulePlannerNode(Node):
    def __init__(self) -> None:
        super().__init__("hybrid_rule_planner")

        self.declare_parameter("rate_hz", 20.0)
        self.declare_parameter("goal_x", 8.0)
        self.declare_parameter("goal_y", 0.0)
        self.declare_parameter("reach_tolerance", 0.4)
        self.declare_parameter("max_linear_speed", 0.8)
        self.declare_parameter("max_angular_speed", 0.7)
        self.declare_parameter("k_linear", 0.8)
        self.declare_parameter("k_heading", 1.5)
        self.declare_parameter("scan_topic", "/scan")
        self.declare_parameter("avoid_distance", 1.0)
        self.declare_parameter("avoid_gain", 1.2)
        self.declare_parameter("reverse_enabled", True)
        self.declare_parameter("reverse_trigger_distance", 0.28)
        self.declare_parameter("reverse_speed", 0.30)
        self.declare_parameter("reverse_turn_speed", 0.55)
        self.declare_parameter("reverse_duration_sec", 1.2)
        self.declare_parameter("escape_forward_duration_sec", 1.0)
        self.declare_parameter("escape_forward_speed", 0.30)
        self.declare_parameter("reverse_turn_mode", "auto")
        self.declare_parameter("output_topic", "/cmd_vel_raw")
        self.declare_parameter("odom_topic", "/odom")
        self.declare_parameter("use_episode_events", True)
        self.declare_parameter("episode_event_topic", "/hybrid_nav/episode_event")

        self.rate_hz = float(self.get_parameter("rate_hz").value)
        self.goal_x = float(self.get_parameter("goal_x").value)
        self.goal_y = float(self.get_parameter("goal_y").value)
        self.reach_tolerance = float(self.get_parameter("reach_tolerance").value)
        self.max_linear_speed = float(self.get_parameter("max_linear_speed").value)
        self.max_angular_speed = float(self.get_parameter("max_angular_speed").value)
        self.k_linear = float(self.get_parameter("k_linear").value)
        self.k_heading = float(self.get_parameter("k_heading").value)
        self.scan_topic = str(self.get_parameter("scan_topic").value)
        self.avoid_distance = float(self.get_parameter("avoid_distance").value)
        self.avoid_gain = float(self.get_parameter("avoid_gain").value)
        self.reverse_enabled = bool(self.get_parameter("reverse_enabled").value)
        self.reverse_trigger_distance = float(self.get_parameter("reverse_trigger_distance").value)
        self.reverse_speed = float(self.get_parameter("reverse_speed").value)
        self.reverse_turn_speed = float(self.get_parameter("reverse_turn_speed").value)
        self.reverse_duration_sec = float(self.get_parameter("reverse_duration_sec").value)
        self.escape_forward_duration_sec = float(
            self.get_parameter("escape_forward_duration_sec").value
        )
        self.escape_forward_speed = float(self.get_parameter("escape_forward_speed").value)
        self.reverse_turn_mode = str(self.get_parameter("reverse_turn_mode").value).strip().lower()
        if self.reverse_turn_mode not in ("auto", "left", "right"):
            self.get_logger().warn(
                f"Unknown reverse_turn_mode='{self.reverse_turn_mode}', using 'auto'"
            )
            self.reverse_turn_mode = "auto"
        self.output_topic = str(self.get_parameter("output_topic").value)
        self.odom_topic = str(self.get_parameter("odom_topic").value)
        self.use_episode_events = bool(self.get_parameter("use_episode_events").value)
        self.episode_event_topic = str(self.get_parameter("episode_event_topic").value)

        self.pose_x = 0.0
        self.pose_y = 0.0
        self.yaw = 0.0
        self.odom_received = False
        self.front_range = math.inf
        self.left_range = math.inf
        self.right_range = math.inf
        self.scan_min_range = math.inf
        self.escape_phase = "none"
        self.escape_until_sec = 0.0
        self.escape_turn_sign = 1.0

        self.create_subscription(Odometry, self.odom_topic, self.on_odom, 10)
        self.create_subscription(LaserScan, self.scan_topic, self.on_scan, qos_profile_sensor_data)
        if self.use_episode_events:
            self.create_subscription(String, self.episode_event_topic, self.on_episode_event, 20)
        self.cmd_pub = self.create_publisher(Twist, self.output_topic, 10)
        period = 1.0 / max(self.rate_hz, 1.0)
        self.create_timer(period, self.on_tick)

        self.get_logger().info(
            f"Rule planner started: goal=({self.goal_x:.2f}, {self.goal_y:.2f}), "
            f"output={self.output_topic}"
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

        now_sec = self._now_sec()
        if self._in_escape(now_sec):
            cmd = self._compute_escape_cmd(now_sec)
            self.cmd_pub.publish(cmd)
            return

        if self._should_start_escape():
            self._start_escape(now_sec)
            cmd = self._compute_escape_cmd(now_sec)
            self.cmd_pub.publish(cmd)
            return

        heading_target = math.atan2(dy, dx)
        heading_error = self._wrap_to_pi(heading_target - self.yaw)
        avoidance_turn = self._compute_avoidance_turn()
        speed_scale = self._compute_speed_scale()

        cmd.linear.x = max(-self.max_linear_speed, min(self.max_linear_speed, self.k_linear * dist))
        cmd.linear.x *= speed_scale
        cmd.angular.z = max(
            -self.max_angular_speed,
            min(self.max_angular_speed, self.k_heading * heading_error + avoidance_turn),
        )
        self.cmd_pub.publish(cmd)

    def on_scan(self, msg: LaserScan) -> None:
        front_min = math.inf
        left_min = math.inf
        right_min = math.inf
        scan_min = math.inf

        for i, r in enumerate(msg.ranges):
            if not math.isfinite(r):
                continue
            if r < msg.range_min or r > msg.range_max:
                continue
            scan_min = min(scan_min, r)
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
        self.scan_min_range = scan_min

    def _compute_avoidance_turn(self) -> float:
        if not math.isfinite(self.front_range) or self.front_range >= self.avoid_distance:
            return 0.0
        deficit = (self.avoid_distance - self.front_range) / max(self.avoid_distance, 1e-3)
        turn_mag = self.avoid_gain * deficit
        # Turn away from the closer side obstacle.
        if self.left_range < self.right_range:
            return -turn_mag
        return turn_mag

    def _compute_speed_scale(self) -> float:
        closest = min(self.front_range, self.scan_min_range)
        if not math.isfinite(closest) or closest >= self.avoid_distance:
            return 1.0
        ratio = closest / max(self.avoid_distance, 1e-3)
        return self._clamp(ratio, 0.25, 1.0)

    def _should_start_escape(self) -> bool:
        if not self.reverse_enabled or self.escape_phase != "none":
            return False
        closest = min(self.front_range, self.scan_min_range)
        if not math.isfinite(closest):
            return False
        return closest <= self.reverse_trigger_distance

    def _start_escape(self, now_sec: float) -> None:
        self.escape_phase = "reverse"
        self.escape_until_sec = now_sec + max(0.1, self.reverse_duration_sec)
        self.escape_turn_sign = self._pick_turn_sign()
        closest = min(self.front_range, self.scan_min_range)
        self.get_logger().warn(
            f"Blocked ({closest:.2f} m). Escape started, turn_mode={self.reverse_turn_mode}."
        )

    def _in_escape(self, now_sec: float) -> bool:
        if self.escape_phase == "none":
            return False
        if now_sec < self.escape_until_sec:
            return True
        if self.escape_phase == "reverse" and self.escape_forward_duration_sec > 0.0:
            self.escape_phase = "forward"
            self.escape_until_sec = now_sec + self.escape_forward_duration_sec
            return True
        self.escape_phase = "none"
        return False

    def _compute_escape_cmd(self, now_sec: float) -> Twist:
        cmd = Twist()
        if not self._in_escape(now_sec):
            return cmd

        if self.escape_phase == "reverse":
            cmd.linear.x = -self._clamp(self.reverse_speed, 0.0, self.max_linear_speed)
            cmd.angular.z = self._clamp(
                -self.escape_turn_sign * self.reverse_turn_speed,
                -self.max_angular_speed,
                self.max_angular_speed,
            )
        else:
            cmd.linear.x = self._clamp(self.escape_forward_speed, 0.0, self.max_linear_speed)
            cmd.angular.z = self._clamp(
                self.escape_turn_sign * self.reverse_turn_speed,
                -self.max_angular_speed,
                self.max_angular_speed,
            )
        return cmd

    def _pick_turn_sign(self) -> float:
        if self.reverse_turn_mode == "left":
            return 1.0
        if self.reverse_turn_mode == "right":
            return -1.0
        # Auto: pick side with more free space.
        if self.left_range >= self.right_range:
            return 1.0
        return -1.0

    def _now_sec(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    @staticmethod
    def _clamp(value: float, min_v: float, max_v: float) -> float:
        return max(min_v, min(max_v, value))

    @staticmethod
    def _wrap_to_pi(angle: float) -> float:
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle


def main() -> None:
    rclpy.init()
    node = RulePlannerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
