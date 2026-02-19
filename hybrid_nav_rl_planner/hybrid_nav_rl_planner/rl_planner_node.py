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
        self.declare_parameter("default_k_front_brake", 1.2)
        self.declare_parameter("default_k_heading_rate_limit", 1.2)
        self.declare_parameter("heading_rate_limit_min", 1.0)
        self.declare_parameter("heading_rate_limit_max", 6.0)
        self.declare_parameter("avoid_distance", 1.1)
        self.declare_parameter("front_brake_gain_cap", 0.55)
        self.declare_parameter("min_linear_unblocked", 0.14)
        self.declare_parameter("min_linear_heading_gate_rad", 0.90)
        self.declare_parameter("linear_floor_front_clearance", 0.60)
        self.declare_parameter("standstill_front_clearance", 0.50)
        self.declare_parameter("standstill_heading_gate_rad", 1.05)
        self.declare_parameter("standstill_nudge_linear", 0.08)
        self.declare_parameter("standstill_nudge_delay_sec", 1.2)
        self.declare_parameter("escape_assist_enabled", True)
        self.declare_parameter("escape_front_threshold", 0.32)
        self.declare_parameter("escape_turn_gain", 0.35)
        self.declare_parameter("escape_stagnation_sec", 2.0)
        self.declare_parameter("escape_min_linear", 0.06)
        self.declare_parameter("reverse_escape_enabled", True)
        self.declare_parameter("reverse_escape_front_threshold", 0.32)
        self.declare_parameter("reverse_escape_hard_front_threshold", 0.24)
        self.declare_parameter("reverse_escape_stagnation_sec", 0.4)
        self.declare_parameter("reverse_escape_duration_sec", 1.6)
        self.declare_parameter("reverse_escape_speed", 0.36)
        self.declare_parameter("reverse_escape_turn_speed", 0.70)
        self.declare_parameter("reverse_escape_cooldown_sec", 0.3)
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
        self.k_front_brake = float(self.get_parameter("default_k_front_brake").value)
        self.k_heading_rate_limit = float(
            self.get_parameter("default_k_heading_rate_limit").value
        )
        self.heading_rate_limit_min = float(
            self.get_parameter("heading_rate_limit_min").value
        )
        self.heading_rate_limit_max = float(
            self.get_parameter("heading_rate_limit_max").value
        )
        self.avoid_distance = float(self.get_parameter("avoid_distance").value)
        self.front_brake_gain_cap = float(self.get_parameter("front_brake_gain_cap").value)
        self.min_linear_unblocked = float(self.get_parameter("min_linear_unblocked").value)
        self.min_linear_heading_gate_rad = float(
            self.get_parameter("min_linear_heading_gate_rad").value
        )
        self.linear_floor_front_clearance = float(
            self.get_parameter("linear_floor_front_clearance").value
        )
        self.standstill_front_clearance = float(
            self.get_parameter("standstill_front_clearance").value
        )
        self.standstill_heading_gate_rad = float(
            self.get_parameter("standstill_heading_gate_rad").value
        )
        self.standstill_nudge_linear = float(
            self.get_parameter("standstill_nudge_linear").value
        )
        self.standstill_nudge_delay_sec = float(
            self.get_parameter("standstill_nudge_delay_sec").value
        )
        self.escape_assist_enabled = bool(self.get_parameter("escape_assist_enabled").value)
        self.escape_front_threshold = float(self.get_parameter("escape_front_threshold").value)
        self.escape_turn_gain = float(self.get_parameter("escape_turn_gain").value)
        self.escape_stagnation_sec = float(self.get_parameter("escape_stagnation_sec").value)
        self.escape_min_linear = float(self.get_parameter("escape_min_linear").value)
        self.reverse_escape_enabled = bool(self.get_parameter("reverse_escape_enabled").value)
        self.reverse_escape_front_threshold = float(
            self.get_parameter("reverse_escape_front_threshold").value
        )
        self.reverse_escape_hard_front_threshold = float(
            self.get_parameter("reverse_escape_hard_front_threshold").value
        )
        self.reverse_escape_stagnation_sec = float(
            self.get_parameter("reverse_escape_stagnation_sec").value
        )
        self.reverse_escape_duration_sec = float(
            self.get_parameter("reverse_escape_duration_sec").value
        )
        self.reverse_escape_speed = float(self.get_parameter("reverse_escape_speed").value)
        self.reverse_escape_turn_speed = float(
            self.get_parameter("reverse_escape_turn_speed").value
        )
        self.reverse_escape_cooldown_sec = float(
            self.get_parameter("reverse_escape_cooldown_sec").value
        )
        self.use_episode_events = bool(self.get_parameter("use_episode_events").value)
        self.episode_event_topic = str(self.get_parameter("episode_event_topic").value)

        self.pose_x = 0.0
        self.pose_y = 0.0
        self.yaw = 0.0
        self.odom_received = False
        self.front_range = math.inf
        self.left_range = math.inf
        self.right_range = math.inf
        self.prev_goal_dist: Optional[float] = None
        self.last_progress_time_sec = self._now_sec()
        self.prev_cmd_angular = 0.0
        self.last_tick_sec = self._now_sec()
        self.reverse_escape_until_sec = 0.0
        self.reverse_escape_cooldown_until_sec = 0.0
        self.reverse_escape_turn_sign = 1.0

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
        self.prev_goal_dist = None
        self.last_progress_time_sec = self._now_sec()
        self.prev_cmd_angular = 0.0
        self.reverse_escape_until_sec = 0.0
        self.reverse_escape_cooldown_until_sec = 0.0

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
        self.k_front_brake = self._clamp(
            float(data.get("k_front_brake", self.k_front_brake)),
            0.0,
            self.front_brake_gain_cap,
        )
        self.k_heading_rate_limit = float(
            self._clamp(
                float(data.get("k_heading_rate_limit", self.k_heading_rate_limit)),
                self.heading_rate_limit_min,
                self.heading_rate_limit_max,
            )
        )
        self.avoid_distance = float(data.get("avoid_distance", self.avoid_distance))
        self._policy_mtime = mtime
        self.get_logger().info(
            f"Loaded policy: k_linear={self.k_linear:.3f}, "
            f"k_heading={self.k_heading:.3f}, k_avoid={self.k_avoid:.3f}, "
            f"k_front_brake={self.k_front_brake:.3f}, "
            f"k_heading_rate_limit={self.k_heading_rate_limit:.3f}, "
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
        now_sec = self._now_sec()
        dt = max(1e-3, now_sec - self.last_tick_sec)
        self.last_tick_sec = now_sec
        if not self.odom_received:
            self.prev_cmd_angular = 0.0
            self.cmd_pub.publish(cmd)
            return

        dx = self.goal_x - self.pose_x
        dy = self.goal_y - self.pose_y
        dist = math.hypot(dx, dy)
        self._update_progress_tracking(dist, now_sec)
        if dist < self.reach_tolerance:
            self.prev_cmd_angular = 0.0
            self.cmd_pub.publish(cmd)
            return

        heading_target = math.atan2(dy, dx)
        heading_error = self._wrap_to_pi(heading_target - self.yaw)
        avoid_feature = self._compute_avoid_feature()
        front_brake = self._front_brake_feature()

        if self._reverse_escape_active(now_sec):
            self._publish_reverse_escape_cmd()
            return
        if self._should_start_reverse_escape(now_sec):
            self._start_reverse_escape(now_sec)
            self._publish_reverse_escape_cmd()
            return

        linear = self.k_linear * dist * self._front_speed_scale()
        linear -= self.k_front_brake * front_brake
        angular_raw = self.k_heading * heading_error + self.k_avoid * avoid_feature

        if self._escape_assist_active(now_sec):
            linear = max(linear * 0.4, self.escape_min_linear)
            angular_raw += self.escape_turn_gain * self._preferred_turn_sign()

        if self._can_apply_linear_floor(heading_error):
            linear = max(linear, self.min_linear_unblocked)

        if self._needs_forward_nudge(now_sec, heading_error, linear):
            linear = max(linear, self.min_linear_unblocked + self.standstill_nudge_linear)
            angular_raw *= 0.75

        angular = self._rate_limit_angular(angular_raw, dt)

        cmd.linear.x = self._clamp(linear, 0.0, self.max_linear_speed)
        cmd.angular.z = self._clamp(angular, -self.max_angular_speed, self.max_angular_speed)
        self.prev_cmd_angular = cmd.angular.z
        self.cmd_pub.publish(cmd)

    def _front_speed_scale(self) -> float:
        if not math.isfinite(self.front_range):
            return 1.0
        ratio = self.front_range / max(self.avoid_distance, 1e-3)
        return self._clamp(ratio, 0.2, 1.0)

    def _front_brake_feature(self) -> float:
        if not math.isfinite(self.front_range):
            return 0.0
        return self._clamp(
            (self.avoid_distance - self.front_range) / max(self.avoid_distance, 1e-3),
            0.0,
            1.0,
        )

    def _compute_avoid_feature(self) -> float:
        left = self.left_range if math.isfinite(self.left_range) else self.avoid_distance
        right = self.right_range if math.isfinite(self.right_range) else self.avoid_distance
        diff = left - right
        return self._clamp(diff, -1.0, 1.0)

    def _can_apply_linear_floor(self, heading_error: float) -> bool:
        if math.isfinite(self.front_range) and self.front_range <= self.linear_floor_front_clearance:
            return False
        return abs(heading_error) <= self.min_linear_heading_gate_rad

    def _rate_limit_angular(self, angular_raw: float, dt: float) -> float:
        max_delta = max(0.02, self.k_heading_rate_limit * dt)
        limited = self._clamp(
            angular_raw,
            self.prev_cmd_angular - max_delta,
            self.prev_cmd_angular + max_delta,
        )
        return self._clamp(limited, -self.max_angular_speed, self.max_angular_speed)

    def _update_progress_tracking(self, dist: float, now_sec: float) -> None:
        if self.prev_goal_dist is None:
            self.prev_goal_dist = dist
            self.last_progress_time_sec = now_sec
            return
        if dist < (self.prev_goal_dist - 0.01):
            self.last_progress_time_sec = now_sec
        self.prev_goal_dist = dist

    def _needs_forward_nudge(self, now_sec: float, heading_error: float, linear: float) -> bool:
        if linear >= (0.8 * self.min_linear_unblocked):
            return False
        if abs(heading_error) > self.standstill_heading_gate_rad:
            return False
        if math.isfinite(self.front_range) and self.front_range <= self.standstill_front_clearance:
            return False
        return (now_sec - self.last_progress_time_sec) >= self.standstill_nudge_delay_sec

    def _escape_assist_active(self, now_sec: float) -> bool:
        if not self.escape_assist_enabled:
            return False
        if not math.isfinite(self.front_range):
            return False
        no_progress = (now_sec - self.last_progress_time_sec) >= self.escape_stagnation_sec
        return self.front_range <= self.escape_front_threshold and no_progress

    def _reverse_escape_active(self, now_sec: float) -> bool:
        if not self.reverse_escape_enabled:
            return False
        return now_sec < self.reverse_escape_until_sec

    def _should_start_reverse_escape(self, now_sec: float) -> bool:
        if not self.reverse_escape_enabled:
            return False
        if now_sec < self.reverse_escape_until_sec:
            return False
        if now_sec < self.reverse_escape_cooldown_until_sec:
            return False
        if not math.isfinite(self.front_range):
            return False
        if self.front_range <= self.reverse_escape_hard_front_threshold:
            return True
        if self.front_range > self.reverse_escape_front_threshold:
            return False
        return (now_sec - self.last_progress_time_sec) >= self.reverse_escape_stagnation_sec

    def _start_reverse_escape(self, now_sec: float) -> None:
        duration = max(0.2, self.reverse_escape_duration_sec)
        self.reverse_escape_until_sec = now_sec + duration
        self.reverse_escape_cooldown_until_sec = (
            self.reverse_escape_until_sec + max(0.0, self.reverse_escape_cooldown_sec)
        )
        self.reverse_escape_turn_sign = self._preferred_turn_sign()
        self.get_logger().warn(
            f"Reverse escape triggered: front_range={self.front_range:.2f} m, "
            f"duration={duration:.2f}s"
        )

    def _publish_reverse_escape_cmd(self) -> None:
        cmd = Twist()
        cmd.linear.x = -self._clamp(self.reverse_escape_speed, 0.0, self.max_linear_speed)
        cmd.angular.z = self._clamp(
            -self.reverse_escape_turn_sign * self.reverse_escape_turn_speed,
            -self.max_angular_speed,
            self.max_angular_speed,
        )
        self.prev_cmd_angular = cmd.angular.z
        self.cmd_pub.publish(cmd)

    def _preferred_turn_sign(self) -> float:
        left = self.left_range if math.isfinite(self.left_range) else self.avoid_distance
        right = self.right_range if math.isfinite(self.right_range) else self.avoid_distance
        if left >= right:
            return 1.0
        return -1.0

    def _now_sec(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

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
