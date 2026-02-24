#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter, deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import rclpy
from rcl_interfaces.msg import Log
from rclpy.node import Node
from std_msgs.msg import String


def _now_wall() -> str:
    return datetime.now().strftime("%H:%M:%S")


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


@dataclass(slots=True)
class GoalSample:
    x: float
    y: float
    source: str
    stamp_sec: float


class SemanticDiagMonitor(Node):
    def __init__(
        self,
        summary_period_sec: float,
        output_path: Path,
        goal_history: int,
        pingpong_tol_m: float,
    ) -> None:
        super().__init__("semantic_diag_monitor")
        self.summary_period_sec = max(1.0, summary_period_sec)
        self.output_path = output_path
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.goal_history = max(6, goal_history)
        self.pingpong_tol_m = max(0.01, pingpong_tol_m)

        self.status_counts: Counter[str] = Counter()
        self.issue_counts: Counter[str] = Counter()
        self.node_counts: Counter[str] = Counter()
        self.scoring_mode_counts: Counter[str] = Counter()
        self.goal_samples: deque[GoalSample] = deque(maxlen=self.goal_history)
        self.recent_issues: deque[str] = deque(maxlen=8)
        self.stuck_hotspots: Counter[str] = Counter()
        self.start_mono = time.monotonic()

        self._event_file = self.output_path.open("a", encoding="utf-8")

        self.create_subscription(String, "/exploration/status", self.on_exploration_status, 50)
        self.create_subscription(String, "/semantic_rl/candidate_scores_json", self.on_candidate_scores, 20)
        self.create_subscription(Log, "/rosout", self.on_rosout, 300)
        self.create_timer(self.summary_period_sec, self.print_summary)

        self.get_logger().info(
            f"semantic_diag_monitor started: summary={self.summary_period_sec:.1f}s output={self.output_path}"
        )

    def close(self) -> None:
        if not self._event_file.closed:
            self._event_file.close()

    def on_exploration_status(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data)
        except json.JSONDecodeError:
            self.issue_counts["status_json_decode_error"] += 1
            return

        event = str(payload.get("event", "unknown"))
        self.status_counts[event] += 1
        self._write_event("status", payload)

        if event == "goal_sent":
            sample = GoalSample(
                x=_safe_float(payload.get("target_x")),
                y=_safe_float(payload.get("target_y")),
                source=str(payload.get("source", "unknown")),
                stamp_sec=_safe_float(payload.get("stamp_sec")),
            )
            self.goal_samples.append(sample)
        elif event in ("stuck", "timeout", "goal_aborted"):
            hx, hy = self._quantize_xy(
                _safe_float(payload.get("target_x"), float("nan")),
                _safe_float(payload.get("target_y"), float("nan")),
            )
            if hx is not None and hy is not None:
                self.stuck_hotspots[f"{hx:.2f},{hy:.2f}"] += 1

    def on_rosout(self, msg: Log) -> None:
        node_name = str(msg.name)
        message = str(msg.msg)
        self.node_counts[node_name] += 1

        matched_issue: str | None = None
        lower = message.lower()
        if "message filter dropping message" in lower:
            matched_issue = "message_filter_drop"
        elif "failed to create plan with tolerance" in lower:
            matched_issue = "planner_failed_to_plan"
        elif "extrapolation into the future" in lower:
            matched_issue = "tf_extrapolation_future"
        elif "earlier than all the data in the transform cache" in lower:
            matched_issue = "tf_cache_earlier_than_transform"
        elif "unable to transform goal pose into costmap frame" in lower:
            matched_issue = "goal_transform_failed"
        elif "no progress detected. cancelling current navigation goal" in lower:
            matched_issue = "no_progress_cancel"
        elif "running spin" in lower:
            matched_issue = "recovery_spin_start"
        elif "canceling spin" in lower:
            matched_issue = "recovery_spin_cancel"
        elif "spin failed" in lower:
            matched_issue = "spin_failed"
        elif "backup failed" in lower:
            matched_issue = "backup_failed"
        elif "no goal published: all candidates filtered" in lower:
            matched_issue = "all_candidates_filtered"

        if matched_issue is not None:
            self.issue_counts[matched_issue] += 1
            snippet = f"{node_name}: {message[:170]}"
            self.recent_issues.append(snippet)
            self._write_event(
                "rosout_issue",
                {
                    "node": node_name,
                    "level": int(msg.level),
                    "issue": matched_issue,
                    "message": message,
                    "stamp_sec": _safe_float(msg.stamp.sec) + _safe_float(msg.stamp.nanosec) * 1e-9,
                },
            )

    def on_candidate_scores(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data)
        except json.JSONDecodeError:
            self.issue_counts["scores_json_decode_error"] += 1
            return
        scoring_mode = str(payload.get("scoring_mode", "unknown"))
        self.scoring_mode_counts[scoring_mode] += 1

    def _write_event(self, kind: str, payload: dict[str, Any]) -> None:
        row = {
            "wall_time": datetime.now().isoformat(timespec="milliseconds"),
            "kind": kind,
            "payload": payload,
        }
        self._event_file.write(json.dumps(row, ensure_ascii=True) + "\n")
        self._event_file.flush()

    def _detect_goal_pingpong(self) -> tuple[bool, str]:
        samples = list(self.goal_samples)
        if len(samples) < 6:
            return False, ""

        centers: list[tuple[float, float]] = []
        labels: list[int] = []
        for s in samples:
            idx = -1
            for i, c in enumerate(centers):
                if math.hypot(s.x - c[0], s.y - c[1]) <= self.pingpong_tol_m:
                    idx = i
                    break
            if idx == -1:
                centers.append((s.x, s.y))
                idx = len(centers) - 1
            labels.append(idx)

        if len(centers) != 2:
            return False, ""

        alternations = 0
        for i in range(1, len(labels)):
            if labels[i] != labels[i - 1]:
                alternations += 1
        if alternations < (len(labels) - 2):
            return False, ""

        a = centers[0]
        b = centers[1]
        detail = f"A=({a[0]:.2f},{a[1]:.2f}) <-> B=({b[0]:.2f},{b[1]:.2f})"
        return True, detail

    @staticmethod
    def _quantize_xy(x: float, y: float, step: float = 0.25) -> tuple[float | None, float | None]:
        if not math.isfinite(x) or not math.isfinite(y):
            return None, None
        qx = round(x / step) * step
        qy = round(y / step) * step
        return qx, qy

    def _top_stuck_hotspots(self, limit: int = 5) -> list[dict[str, Any]]:
        if not self.stuck_hotspots:
            return []
        out: list[dict[str, Any]] = []
        for key, count in self.stuck_hotspots.most_common(limit):
            sx, sy = key.split(",", 1)
            out.append({"x": float(sx), "y": float(sy), "count": int(count)})
        return out

    def _dominant_failure_mode(self) -> str:
        keys = (
            "planner_failed_to_plan",
            "tf_extrapolation_future",
            "goal_transform_failed",
            "message_filter_drop",
            "all_candidates_filtered",
            "spin_failed",
            "backup_failed",
            "no_progress_cancel",
        )
        best_key = ""
        best_val = 0
        for key in keys:
            val = int(self.issue_counts[key])
            if val > best_val:
                best_val = val
                best_key = key
        return best_key if best_key else "none"

    def print_summary(self) -> None:
        uptime = time.monotonic() - self.start_mono
        pingpong, pingpong_detail = self._detect_goal_pingpong()
        if pingpong:
            self.issue_counts["goal_pingpong"] += 1

        status = self.status_counts
        issues = self.issue_counts
        line = (
            f"[{_now_wall()} +{uptime:6.1f}s] "
            f"goal_sent={status['goal_sent']} succ={status['goal_succeeded']} "
            f"abort={status['goal_aborted']} cancel={status['goal_canceled']} "
            f"stuck={status['stuck']} timeout={status['timeout']} | "
            f"mode(rule/policy)={self.scoring_mode_counts['rule']}/{self.scoring_mode_counts['policy']} | "
            f"tf_extrap={issues['tf_extrapolation_future']} "
            f"plan_fail={issues['planner_failed_to_plan']} "
            f"filter_drop={issues['message_filter_drop']} "
            f"transform_fail={issues['goal_transform_failed']} "
            f"no_goal={issues['all_candidates_filtered']} "
            f"spin={issues['recovery_spin_start']}/{issues['recovery_spin_cancel']} "
            f"spin_fail={issues['spin_failed']} backup_fail={issues['backup_failed']}"
        )
        print(line, flush=True)
        if pingpong:
            print(f"  PINGPONG detected: {pingpong_detail}", flush=True)
        if self.recent_issues:
            print(f"  Last issue: {self.recent_issues[-1]}", flush=True)

    def print_final_report(self) -> None:
        pingpong, pingpong_detail = self._detect_goal_pingpong()
        report = {
            "wall_time": datetime.now().isoformat(timespec="seconds"),
            "uptime_sec": round(time.monotonic() - self.start_mono, 3),
            "status_counts": dict(self.status_counts),
            "issue_counts": dict(self.issue_counts),
            "scoring_mode_counts": dict(self.scoring_mode_counts),
            "planner_failed_to_plan_count": int(self.issue_counts["planner_failed_to_plan"]),
            "dominant_failure_mode": self._dominant_failure_mode(),
            "stuck_hotspots": self._top_stuck_hotspots(),
            "pingpong_detected": pingpong,
            "pingpong_detail": pingpong_detail,
            "recent_issues": list(self.recent_issues),
        }
        summary_path = self.output_path.with_suffix(".summary.json")
        summary_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nFinal summary saved: {summary_path}", flush=True)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Live semantic autonomy diagnostics monitor")
    parser.add_argument(
        "--summary-period",
        type=float,
        default=5.0,
        help="Periodic summary print interval in seconds",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/tmp/semantic_diag_events.jsonl"),
        help="Output JSONL path for compact diagnostics events",
    )
    parser.add_argument(
        "--goal-history",
        type=int,
        default=12,
        help="Goal history window for ping-pong detection",
    )
    parser.add_argument(
        "--pingpong-tol",
        type=float,
        default=0.35,
        help="Distance tolerance (m) to cluster goals for ping-pong detection",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    rclpy.init()
    node = SemanticDiagMonitor(
        summary_period_sec=args.summary_period,
        output_path=args.output,
        goal_history=args.goal_history,
        pingpong_tol_m=args.pingpong_tol,
    )
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.print_final_report()
        node.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
