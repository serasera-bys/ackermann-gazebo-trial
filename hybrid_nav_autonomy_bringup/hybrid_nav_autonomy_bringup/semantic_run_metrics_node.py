from __future__ import annotations

import json
import os
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import rclpy
from nav_msgs.msg import OccupancyGrid
from rclpy.node import Node
from std_msgs.msg import String
from vision_msgs.msg import Detection2DArray


def _detect_repo_root() -> Path:
    env_root = os.environ.get("HYBRID_NAV_ROBOT_ROOT", "").strip()
    if env_root:
        path = Path(env_root).expanduser().resolve()
        if path.exists():
            return path

    here = Path(__file__).resolve()
    for parent in [here, *here.parents]:
        if (parent / "experiments").exists():
            return parent

    for parent in here.parents:
        if parent.name == "install":
            candidate = parent.parent / "src" / "hybrid_nav_robot"
            if candidate.exists():
                return candidate

    return Path.cwd()


def _default_output_dir() -> str:
    return str(_detect_repo_root() / "experiments" / "results" / "semantic_explorer")


class SemanticRunMetricsNode(Node):
    def __init__(self) -> None:
        super().__init__("semantic_run_metrics")

        self.declare_parameter("map_topic", "/map")
        self.declare_parameter("detections_topic", "/semantic/detections")
        self.declare_parameter("status_topic", "/exploration/status")
        self.declare_parameter("output_dir", _default_output_dir())
        self.declare_parameter("write_period_sec", 2.0)
        self.declare_parameter("fps_window_sec", 5.0)

        map_topic = str(self.get_parameter("map_topic").value)
        detections_topic = str(self.get_parameter("detections_topic").value)
        status_topic = str(self.get_parameter("status_topic").value)
        output_dir = Path(str(self.get_parameter("output_dir").value))
        write_period_sec = max(0.5, float(self.get_parameter("write_period_sec").value))
        self.fps_window_sec = max(1.0, float(self.get_parameter("fps_window_sec").value))

        run_id = datetime.now(timezone.utc).strftime("run_%Y%m%d_%H%M%S")
        self.run_dir = output_dir / run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.run_dir / "metrics.json"

        self.start_sec = self._now_sec()
        self.latest_map: OccupancyGrid | None = None
        self.event_counts: dict[str, int] = {}
        self.detection_times_sec: deque[float] = deque()

        self.create_subscription(OccupancyGrid, map_topic, self.on_map, 10)
        self.create_subscription(Detection2DArray, detections_topic, self.on_detections, 20)
        self.create_subscription(String, status_topic, self.on_status, 20)
        self.create_timer(write_period_sec, self.on_timer)

        self.get_logger().info(f"semantic_run_metrics writing to {self.metrics_path}")

    def on_map(self, msg: OccupancyGrid) -> None:
        self.latest_map = msg

    def on_detections(self, _msg: Detection2DArray) -> None:
        now = self._now_sec()
        self.detection_times_sec.append(now)
        self._trim_detection_times(now)

    def on_status(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        event = str(payload.get("event", "")).strip()
        if not event:
            return
        self.event_counts[event] = self.event_counts.get(event, 0) + 1

    def on_timer(self) -> None:
        now = self._now_sec()
        self._trim_detection_times(now)
        map_metrics = self._map_metrics()

        payload = {
            "run_id": self.run_dir.name,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "uptime_sec": now - self.start_sec,
            "detection_fps_avg": self._detection_fps(now),
            "events": dict(self.event_counts),
            "goals_sent": int(self.event_counts.get("goal_sent", 0)),
            "goals_succeeded": int(self.event_counts.get("goal_succeeded", 0)),
            "goals_aborted": int(self.event_counts.get("goal_aborted", 0)),
            "goal_timeouts": int(self.event_counts.get("timeout", 0)),
            "map": map_metrics,
        }
        self.metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _trim_detection_times(self, now_sec: float) -> None:
        cutoff = now_sec - self.fps_window_sec
        while self.detection_times_sec and self.detection_times_sec[0] < cutoff:
            self.detection_times_sec.popleft()

    def _detection_fps(self, now_sec: float) -> float:
        if not self.detection_times_sec:
            return 0.0
        span = max(1e-3, min(self.fps_window_sec, now_sec - self.detection_times_sec[0]))
        return float(len(self.detection_times_sec) / span)

    def _map_metrics(self) -> dict[str, float | int]:
        if self.latest_map is None:
            return {
                "width": 0,
                "height": 0,
                "known_ratio": 0.0,
                "free_ratio": 0.0,
                "occupied_ratio": 0.0,
            }

        data = list(self.latest_map.data)
        total = max(1, len(data))
        known = sum(1 for v in data if v >= 0)
        free = sum(1 for v in data if v == 0)
        occupied = sum(1 for v in data if v > 50)
        return {
            "width": int(self.latest_map.info.width),
            "height": int(self.latest_map.info.height),
            "known_ratio": float(known / total),
            "free_ratio": float(free / total),
            "occupied_ratio": float(occupied / total),
        }

    def destroy_node(self) -> bool:
        try:
            self.on_timer()
        except Exception:
            pass
        return super().destroy_node()

    def _now_sec(self) -> float:
        return self.get_clock().now().nanoseconds / 1e9


def main() -> None:
    rclpy.init()
    node = SemanticRunMetricsNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
