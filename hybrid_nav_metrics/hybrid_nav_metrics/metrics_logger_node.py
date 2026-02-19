import json
import math
from pathlib import Path
from typing import Optional, TextIO

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool, String


def _default_experiment_file(filename: str) -> str:
    base = Path.home() / ".ros" / "hybrid_nav_robot" / "experiments"
    return str(base / filename)


class MetricsLoggerNode(Node):
    def __init__(self) -> None:
        super().__init__("hybrid_metrics_logger")

        self.declare_parameter(
            "output_file",
            _default_experiment_file("latest_metrics.json"),
        )
        self.declare_parameter("odom_topic", "/odom")
        self.declare_parameter("cmd_topic", "/cmd_vel")
        self.declare_parameter("scan_topic", "/scan")
        self.declare_parameter("safety_topic", "/safety_layer/intervention")
        self.declare_parameter("episode_event_topic", "/hybrid_nav/episode_event")
        self.declare_parameter("use_episode_events", True)
        self.declare_parameter(
            "episode_summary_file",
            _default_experiment_file("episode_metrics.jsonl"),
        )
        self.declare_parameter("append_episode_summary", False)
        self.declare_parameter("planner_mode_label", "")
        self.declare_parameter("scenario_label", "")
        self.declare_parameter("goal_x", 8.0)
        self.declare_parameter("goal_y", 0.0)
        self.declare_parameter("goal_tolerance", 0.4)
        self.declare_parameter("collision_distance", 0.2)

        self.output_file = str(self.get_parameter("output_file").value)
        odom_topic = str(self.get_parameter("odom_topic").value)
        cmd_topic = str(self.get_parameter("cmd_topic").value)
        scan_topic = str(self.get_parameter("scan_topic").value)
        safety_topic = str(self.get_parameter("safety_topic").value)
        episode_event_topic = str(self.get_parameter("episode_event_topic").value)
        self.use_episode_events = bool(self.get_parameter("use_episode_events").value)
        self.episode_summary_file = str(self.get_parameter("episode_summary_file").value)
        self.append_episode_summary = bool(self.get_parameter("append_episode_summary").value)
        self.planner_mode_label = str(self.get_parameter("planner_mode_label").value).strip()
        self.scenario_label = str(self.get_parameter("scenario_label").value).strip()
        self.goal_x = float(self.get_parameter("goal_x").value)
        self.goal_y = float(self.get_parameter("goal_y").value)
        self.goal_tolerance = float(self.get_parameter("goal_tolerance").value)
        self.collision_distance = float(self.get_parameter("collision_distance").value)

        self.run_start_time = self.get_clock().now()
        self.start_time = self.run_start_time
        self.episode_id = 0
        self.episode_active = False
        self.episode_end_reason = ""
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
        self._episode_summary_stream: Optional[TextIO] = None

        if self.append_episode_summary:
            summary_path = Path(self.episode_summary_file)
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            self._episode_summary_stream = summary_path.open("a", encoding="utf-8")

        self.create_subscription(Odometry, odom_topic, self.on_odom, 20)
        self.create_subscription(Twist, cmd_topic, self.on_cmd, 20)
        # LaserScan from simulation typically uses SensorDataQoS (best effort).
        self.create_subscription(LaserScan, scan_topic, self.on_scan, qos_profile_sensor_data)
        self.create_subscription(Bool, safety_topic, self.on_safety, 10)
        if self.use_episode_events:
            self.create_subscription(String, episode_event_topic, self.on_episode_event, 20)
        self.create_timer(2.0, self.on_heartbeat)

        rclpy.get_default_context().on_shutdown(self.on_shutdown)
        self.get_logger().info(
            f"Metrics logger started, output={self.output_file}, "
            f"use_episode_events={self.use_episode_events}, "
            f"append_episode_summary={self.append_episode_summary}"
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
            self.episode_active = True
            self.episode_end_reason = ""
            self._reset_episode_metrics()
            self.get_logger().info(
                f"Metrics episode {self.episode_id} started: goal=({self.goal_x:.2f}, {self.goal_y:.2f})"
            )
        elif event == "episode_end":
            self.episode_id = int(payload.get("episode_id", self.episode_id))
            self.episode_active = False
            self.episode_end_reason = str(payload.get("reason", "unknown"))
            if "dist_to_goal" in payload:
                self.goal_distance = float(payload["dist_to_goal"])
            self._write_episode_summary(payload)

    def on_odom(self, msg: Odometry) -> None:
        if not self.use_episode_events and self.episode_id == 0:
            self.episode_id = 1
            self.episode_active = True
            self._reset_episode_metrics()
        if self.use_episode_events and not self.episode_active:
            return

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
        self._write_metrics(runtime)
        self.get_logger().info(
            f"episode={self.episode_id}, runtime={runtime:.1f}s, dist={self.distance_traveled:.2f}m, "
            f"goal_dist={self._safe_range(self.goal_distance):.2f}m, "
            f"min_range={self._safe_range(self.min_obstacle_range):.2f}m, "
            f"safety_hits={self.safety_interventions}, goal_reached={self.goal_reached}, "
            f"active={self.episode_active}"
        )

    def on_shutdown(self) -> None:
        runtime = (self.get_clock().now() - self.start_time).nanoseconds / 1e9
        self._write_metrics(runtime)
        if self._episode_summary_stream is not None:
            self._episode_summary_stream.flush()
            self._episode_summary_stream.close()
            self._episode_summary_stream = None
        self.get_logger().info(f"Saved metrics to {self.output_file}")

    def _write_metrics(self, runtime: float) -> None:
        run_runtime = (self.get_clock().now() - self.run_start_time).nanoseconds / 1e9
        data = {
            "run_runtime_sec": run_runtime,
            "episode_id": self.episode_id,
            "episode_active": self.episode_active,
            "episode_end_reason": self.episode_end_reason,
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

    def _reset_episode_metrics(self) -> None:
        self.start_time = self.get_clock().now()
        self.last_x = None
        self.last_y = None
        self.distance_traveled = 0.0
        self.max_linear_cmd = 0.0
        self.max_angular_cmd = 0.0
        self.min_obstacle_range = math.inf
        self.safety_interventions = 0
        self.odom_samples = 0
        self.goal_distance = math.inf
        self.goal_reached = False
        self.time_to_goal_sec = None
        self.collision_flag = False

    def _write_episode_summary(self, payload: dict) -> None:
        if self._episode_summary_stream is None:
            return
        runtime = (self.get_clock().now() - self.start_time).nanoseconds / 1e9
        dist_to_goal = self.goal_distance
        if "dist_to_goal" in payload:
            dist_to_goal = float(payload["dist_to_goal"])
        reason = str(payload.get("reason", self.episode_end_reason or "unknown"))
        goal_reached = reason == "goal" or bool(self.goal_reached)
        summary = {
            "episode_id": self.episode_id,
            "scenario": self.scenario_label,
            "planner_mode": self.planner_mode_label,
            "reason": reason,
            "goal_reached": goal_reached,
            "runtime_sec": runtime,
            "distance_traveled_m": self.distance_traveled,
            "goal_distance_m": self._safe_range(dist_to_goal),
            "safety_interventions": self.safety_interventions,
            "collision_flag": self.collision_flag,
            "min_obstacle_range_m": self._safe_range(self.min_obstacle_range),
        }
        self._episode_summary_stream.write(json.dumps(summary) + "\n")
        self._episode_summary_stream.flush()

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
