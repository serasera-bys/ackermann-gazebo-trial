import json
import math
import subprocess
from pathlib import Path
from typing import Optional

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String


class DatasetCollectorNode(Node):
    def __init__(self) -> None:
        super().__init__("hybrid_rl_dataset_collector")

        self.declare_parameter(
            "output_file",
            "/home/bernard/ros2_ws/src/hybrid_nav_robot/experiments/rl_dataset.jsonl",
        )
        self.declare_parameter("goal_x", 8.0)
        self.declare_parameter("goal_y", 0.0)
        self.declare_parameter("odom_topic", "/ackermann_steering_controller/odometry")
        self.declare_parameter("scan_topic", "/scan")
        self.declare_parameter("cmd_topic", "/cmd_vel_raw")
        self.declare_parameter("flush_every", 50)
        self.declare_parameter("goal_tolerance", 0.25)
        self.declare_parameter("auto_reset", False)
        self.declare_parameter("reset_on_goal", True)
        self.declare_parameter("reset_on_stuck", True)
        self.declare_parameter("reset_pause_sec", 1.0)
        self.declare_parameter("world_control_service", "/world/default/control")
        self.declare_parameter("reset_mode", "model_only")
        self.declare_parameter("progress_distance", 0.15)
        self.declare_parameter("stuck_timeout_sec", 4.0)
        self.declare_parameter("min_episode_sec", 2.0)
        self.declare_parameter("stuck_min_cmd_linear", 0.10)
        self.declare_parameter("cmd_stale_sec", 1.0)
        self.declare_parameter("post_reset_goal_guard_sec", 1.0)
        self.declare_parameter("write_terminal_sample", True)
        self.declare_parameter("use_episode_events", False)
        self.declare_parameter("episode_event_topic", "/hybrid_nav/episode_event")
        self.declare_parameter("terminal_dedup_distance_eps", 0.05)
        self.declare_parameter("terminal_dedup_time_sec", 3.0)

        self.output_file = str(self.get_parameter("output_file").value)
        self.goal_x = float(self.get_parameter("goal_x").value)
        self.goal_y = float(self.get_parameter("goal_y").value)
        self.odom_topic = str(self.get_parameter("odom_topic").value)
        self.scan_topic = str(self.get_parameter("scan_topic").value)
        self.cmd_topic = str(self.get_parameter("cmd_topic").value)
        self.flush_every = max(1, int(self.get_parameter("flush_every").value))
        self.goal_tolerance = float(self.get_parameter("goal_tolerance").value)
        self.auto_reset = bool(self.get_parameter("auto_reset").value)
        self.reset_on_goal = bool(self.get_parameter("reset_on_goal").value)
        self.reset_on_stuck = bool(self.get_parameter("reset_on_stuck").value)
        self.reset_pause_sec = max(0.0, float(self.get_parameter("reset_pause_sec").value))
        self.world_control_service = str(self.get_parameter("world_control_service").value)
        self.reset_mode = str(self.get_parameter("reset_mode").value).strip().lower()
        if self.reset_mode not in ("all", "model_only", "time_only"):
            self.get_logger().warn(
                f"Unknown reset_mode='{self.reset_mode}', using 'model_only'"
            )
            self.reset_mode = "model_only"
        self.progress_distance = max(0.01, float(self.get_parameter("progress_distance").value))
        self.stuck_timeout_sec = max(0.5, float(self.get_parameter("stuck_timeout_sec").value))
        self.min_episode_sec = max(0.0, float(self.get_parameter("min_episode_sec").value))
        self.stuck_min_cmd_linear = max(0.0, float(self.get_parameter("stuck_min_cmd_linear").value))
        self.cmd_stale_sec = max(0.1, float(self.get_parameter("cmd_stale_sec").value))
        self.post_reset_goal_guard_sec = max(
            0.0, float(self.get_parameter("post_reset_goal_guard_sec").value)
        )
        self.write_terminal_sample = bool(self.get_parameter("write_terminal_sample").value)
        self.use_episode_events = bool(self.get_parameter("use_episode_events").value)
        self.episode_event_topic = str(self.get_parameter("episode_event_topic").value)
        self.terminal_dedup_distance_eps = max(
            0.0, float(self.get_parameter("terminal_dedup_distance_eps").value)
        )
        self.terminal_dedup_time_sec = max(
            0.0, float(self.get_parameter("terminal_dedup_time_sec").value)
        )

        self.pose_x = 0.0
        self.pose_y = 0.0
        self.yaw = 0.0
        self.odom_received = False
        self.odom_msg_count = 0
        self.front_range = math.inf
        self.left_range = math.inf
        self.right_range = math.inf
        self.scan_received = False
        self.latest_linear_cmd = 0.0
        self.latest_angular_cmd = 0.0
        self.last_cmd_time_sec = float("-inf")

        self.episode_id = 0
        self.episode_active = False
        self.episode_done = False
        self.episode_start_time_sec = 0.0
        self.episode_start_odom_count = 0
        self.goal_guard_until_sec = 0.0
        self.last_progress_time_sec = 0.0
        self.progress_anchor_x = 0.0
        self.progress_anchor_y = 0.0
        self.pending_reset = False
        self.reset_ready_at_sec = 0.0
        self.last_terminal_reason = ""
        self.last_terminal_dist = math.inf
        self.last_terminal_time_sec = float("-inf")

        out_path = Path(self.output_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        self._f = out_path.open("a", encoding="utf-8")
        self._count = 0

        self.create_subscription(Odometry, self.odom_topic, self.on_odom, 20)
        self.create_subscription(LaserScan, self.scan_topic, self.on_scan, qos_profile_sensor_data)
        self.create_subscription(Twist, self.cmd_topic, self.on_cmd, 20)
        if self.use_episode_events:
            self.create_subscription(String, self.episode_event_topic, self.on_episode_event, 20)
        self.create_timer(0.2, self.on_episode_timer)

        self.get_logger().info(
            f"Dataset collector started: cmd={self.cmd_topic}, odom={self.odom_topic}, "
            f"scan={self.scan_topic}, output={self.output_file}, "
            f"use_episode_events={self.use_episode_events}"
        )

    def on_episode_event(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data)
        except json.JSONDecodeError:
            self.get_logger().warn("Invalid episode event payload; ignoring.")
            return

        event = str(payload.get("event", "")).strip()
        if event == "episode_start":
            self.episode_id = int(payload.get("episode_id", self.episode_id + 1))
            self.goal_x = float(payload.get("goal_x", self.goal_x))
            self.goal_y = float(payload.get("goal_y", self.goal_y))
            self.goal_tolerance = float(payload.get("goal_tolerance", self.goal_tolerance))
            self._start_episode(external=True)
        elif event == "episode_end":
            if not self.episode_active:
                return
            reason = str(payload.get("reason", "unknown"))
            self._finish_episode(reason, external=True)

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
        self.odom_msg_count += 1
        if not self.episode_active:
            if not self.use_episode_events:
                self._start_episode()
            else:
                return
        if self.episode_done or self.pending_reset:
            return

        moved = math.hypot(self.pose_x - self.progress_anchor_x, self.pose_y - self.progress_anchor_y)
        if moved >= self.progress_distance:
            self.progress_anchor_x = self.pose_x
            self.progress_anchor_y = self.pose_y
            self.last_progress_time_sec = self._now_sec()

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
        self.scan_received = True

    def on_cmd(self, msg: Twist) -> None:
        self.latest_linear_cmd = float(msg.linear.x)
        self.latest_angular_cmd = float(msg.angular.z)
        self.last_cmd_time_sec = self._now_sec()
        if not self.odom_received or not self.episode_active:
            return
        if self.episode_done or self.pending_reset:
            return

        self._write_sample(
            action_linear=self.latest_linear_cmd,
            action_angular=self.latest_angular_cmd,
            done=False,
            terminal_reason="",
        )

    def on_episode_timer(self) -> None:
        if not self.odom_received or not self.episode_active:
            return
        if self.use_episode_events:
            return

        now_sec = self._now_sec()
        if self.pending_reset:
            if now_sec >= self.reset_ready_at_sec:
                self.pending_reset = False
                self._start_episode()
            return
        if self.episode_done:
            return
        if self.odom_msg_count <= self.episode_start_odom_count:
            return
        if now_sec < self.goal_guard_until_sec:
            return

        dist, _ = self._goal_features()
        if dist <= self.goal_tolerance:
            self._finish_episode("goal")
            return
        if self._is_stuck(now_sec):
            self._finish_episode("stuck")
            return

    def _start_episode(self, external: bool = False) -> None:
        now_sec = self._now_sec()
        if not external:
            self.episode_id += 1
        self.episode_active = True
        self.episode_done = False
        self.episode_start_time_sec = now_sec
        self.episode_start_odom_count = self.odom_msg_count
        self.goal_guard_until_sec = now_sec + self.post_reset_goal_guard_sec
        self.last_progress_time_sec = now_sec
        self.progress_anchor_x = self.pose_x
        self.progress_anchor_y = self.pose_y
        self.get_logger().info(
            f"Episode {self.episode_id} started: goal=({self.goal_x:.2f}, {self.goal_y:.2f})"
        )

    def _finish_episode(self, reason: str, external: bool = False) -> None:
        self.episode_done = True
        dist, _ = self._goal_features()
        if self.write_terminal_sample and self._should_write_terminal(reason, dist):
            self._write_sample(
                action_linear=0.0,
                action_angular=0.0,
                done=True,
                terminal_reason=reason,
            )
        self._f.flush()
        self.get_logger().info(
            f"Episode {self.episode_id} finished: reason={reason}, "
            f"dist_to_goal={dist:.3f} m, total_samples={self._count}"
        )
        if external:
            self.episode_active = False
            return

        should_reset = self.auto_reset and (
            (reason == "goal" and self.reset_on_goal) or (reason == "stuck" and self.reset_on_stuck)
        )
        if should_reset:
            if self._reset_world():
                self.pending_reset = True
                self.reset_ready_at_sec = self._now_sec() + self.reset_pause_sec
            else:
                self.get_logger().warn(
                    "Auto reset failed. Run manual reset or restart launch before next episode."
                )

    def _is_stuck(self, now_sec: float) -> bool:
        if (now_sec - self.episode_start_time_sec) < self.min_episode_sec:
            return False
        if (now_sec - self.last_cmd_time_sec) > self.cmd_stale_sec:
            return False
        if abs(self.latest_linear_cmd) < self.stuck_min_cmd_linear:
            return False
        return (now_sec - self.last_progress_time_sec) >= self.stuck_timeout_sec

    def _goal_features(self) -> tuple[float, float]:
        dx = self.goal_x - self.pose_x
        dy = self.goal_y - self.pose_y
        dist = math.hypot(dx, dy)
        heading_target = math.atan2(dy, dx)
        heading_error = self._wrap_to_pi(heading_target - self.yaw)
        return dist, heading_error

    def _write_sample(
        self,
        action_linear: float,
        action_angular: float,
        done: bool,
        terminal_reason: str,
    ) -> None:
        dist, heading_error = self._goal_features()
        sample = {
            "episode_id": self.episode_id,
            "episode_time_sec": self._now_sec() - self.episode_start_time_sec,
            "goal_x": self.goal_x,
            "goal_y": self.goal_y,
            "dist": dist,
            "heading_error": heading_error,
            "front_range": self._safe_range(self.front_range),
            "left_range": self._safe_range(self.left_range),
            "right_range": self._safe_range(self.right_range),
            "action_linear": action_linear,
            "action_angular": action_angular,
            "scan_ok": self.scan_received,
            "done": done,
            "terminal_reason": terminal_reason,
        }
        self._f.write(json.dumps(sample) + "\n")
        self._count += 1
        if done or (self._count % self.flush_every == 0):
            self._f.flush()
        if self._count % self.flush_every == 0:
            self.get_logger().info(f"Recorded {self._count} samples")

    def _should_write_terminal(self, reason: str, dist: float) -> bool:
        now_sec = self._now_sec()
        if reason != self.last_terminal_reason:
            self.last_terminal_reason = reason
            self.last_terminal_dist = dist
            self.last_terminal_time_sec = now_sec
            return True
        if abs(dist - self.last_terminal_dist) > self.terminal_dedup_distance_eps:
            self.last_terminal_dist = dist
            self.last_terminal_time_sec = now_sec
            return True
        if (now_sec - self.last_terminal_time_sec) > self.terminal_dedup_time_sec:
            self.last_terminal_time_sec = now_sec
            return True
        return False

    def _reset_world(self) -> bool:
        if self.reset_mode == "all":
            reset_req = "reset: {all: true}"
        elif self.reset_mode == "time_only":
            reset_req = "reset: {time_only: true}"
        else:
            reset_req = "reset: {model_only: true}"

        cmd = [
            "gz",
            "service",
            "-s",
            self.world_control_service,
            "--reqtype",
            "gz.msgs.WorldControl",
            "--reptype",
            "gz.msgs.Boolean",
            "--timeout",
            "2000",
            "--req",
            reset_req,
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=4.0, check=False)
        except (OSError, subprocess.SubprocessError) as exc:
            self.get_logger().warn(f"Failed to call world reset command: {exc}")
            return False
        if result.returncode != 0:
            stderr = result.stderr.strip()
            self.get_logger().warn(
                "World reset command failed: "
                + (stderr if stderr else f"exit_code={result.returncode}")
            )
            return False
        self.get_logger().info("World reset requested")
        return True

    def _now_sec(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def destroy_node(self) -> bool:
        try:
            self._f.flush()
            self._f.close()
        except OSError:
            pass
        return super().destroy_node()

    @staticmethod
    def _wrap_to_pi(angle: float) -> float:
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle

    @staticmethod
    def _safe_range(value: float) -> float:
        return value if math.isfinite(value) else -1.0


def main() -> None:
    rclpy.init()
    node = DatasetCollectorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
