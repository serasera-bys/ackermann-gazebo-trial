from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

import rclpy
from builtin_interfaces.msg import Time
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid, Odometry
from rcl_interfaces.msg import SetParametersResult
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String
from std_srvs.srv import Trigger

from .scoring_utils import (
    FEATURE_ORDER,
    clamp,
    heading_delta,
    linear_policy_score,
    novelty_from_semantic_grid,
    parse_class_weights,
    priority_from_objects,
    rule_score,
    yaw_from_quaternion,
)


def _default_policy_file() -> str:
    env_root = os.environ.get("HYBRID_NAV_ROBOT_ROOT", "").strip()
    if env_root:
        path = Path(env_root).expanduser().resolve()
        if path.exists():
            return str(path / "experiments" / "semantic_rl_policy.json")

    here = Path(__file__).resolve()
    for parent in [here, *here.parents]:
        if (parent / "experiments").exists():
            return str(parent / "experiments" / "semantic_rl_policy.json")

    for parent in here.parents:
        if parent.name == "install":
            candidate = parent.parent / "src" / "hybrid_nav_robot" / "experiments" / "semantic_rl_policy.json"
            if candidate.parent.exists():
                return str(candidate)

    return str(Path.cwd() / "experiments" / "semantic_rl_policy.json")


def map_index(
    map_x: float,
    map_y: float,
    resolution: float,
    width: int,
    height: int,
    origin_x: float,
    origin_y: float,
) -> int | None:
    gx = int((map_x - origin_x) / max(resolution, 1e-9))
    gy = int((map_y - origin_y) / max(resolution, 1e-9))
    if gx < 0 or gy < 0 or gx >= width or gy >= height:
        return None
    return gy * width + gx


class SemanticRLDeciderNode(Node):
    def __init__(self) -> None:
        super().__init__("semantic_rl_decider")

        self.declare_parameter("frontier_candidates_json_topic", "/exploration/frontier_candidates_json")
        self.declare_parameter("selected_goal_topic", "/semantic_rl/selected_goal")
        self.declare_parameter("candidate_scores_topic", "/semantic_rl/candidate_scores_json")
        self.declare_parameter("semantic_grid_topic", "/semantic/grid")
        self.declare_parameter("semantic_observations_topic", "/semantic/object_observations_json")
        self.declare_parameter("odom_topic", "/ackermann_steering_controller/odometry")
        self.declare_parameter("scan_topic", "/scan")
        self.declare_parameter("policy_file", os.environ.get("SEMANTIC_RL_POLICY_FILE", _default_policy_file()))
        self.declare_parameter("goal_cooldown_sec", 2.0)
        self.declare_parameter("min_goal_distance", 0.75)
        self.declare_parameter("min_robot_goal_distance", 0.8)
        self.declare_parameter("allow_close_goal_fallback", True)
        self.declare_parameter("close_goal_min_distance", 0.18)
        self.declare_parameter("allow_duplicate_goal_fallback", True)
        self.declare_parameter("score_mode", "auto")
        self.declare_parameter("rule_bootstrap_sec", 90.0)
        self.declare_parameter("rule_bootstrap_goal_count", 12)
        self.declare_parameter("class_weights_json", "{\"class_0\": 1.0, \"class_2\": 0.7}")
        self.declare_parameter("avoid_distance", 1.2)
        self.declare_parameter("free_gain_scale", 40.0)
        self.declare_parameter("rule_w1", 1.0)
        self.declare_parameter("rule_w2", 1.0)
        self.declare_parameter("rule_w3", 1.0)
        self.declare_parameter("rule_w4", 1.0)
        self.declare_parameter("rule_w5", 1.0)

        candidates_topic = str(self.get_parameter("frontier_candidates_json_topic").value)
        selected_goal_topic = str(self.get_parameter("selected_goal_topic").value)
        scores_topic = str(self.get_parameter("candidate_scores_topic").value)
        semantic_grid_topic = str(self.get_parameter("semantic_grid_topic").value)
        semantic_obs_topic = str(self.get_parameter("semantic_observations_topic").value)
        odom_topic = str(self.get_parameter("odom_topic").value)
        scan_topic = str(self.get_parameter("scan_topic").value)

        self.policy_file = str(self.get_parameter("policy_file").value)
        self.goal_cooldown_sec = float(self.get_parameter("goal_cooldown_sec").value)
        self.min_goal_distance = float(self.get_parameter("min_goal_distance").value)
        self.min_robot_goal_distance = float(self.get_parameter("min_robot_goal_distance").value)
        self.allow_close_goal_fallback = bool(self.get_parameter("allow_close_goal_fallback").value)
        self.close_goal_min_distance = float(self.get_parameter("close_goal_min_distance").value)
        self.allow_duplicate_goal_fallback = bool(self.get_parameter("allow_duplicate_goal_fallback").value)
        self.score_mode = str(self.get_parameter("score_mode").value).strip().lower()
        if self.score_mode not in ("auto", "rule_only", "policy_only"):
            self.get_logger().warn(f"Unsupported score_mode='{self.score_mode}', falling back to 'auto'")
            self.score_mode = "auto"
        self.rule_bootstrap_sec = max(0.0, float(self.get_parameter("rule_bootstrap_sec").value))
        self.rule_bootstrap_goal_count = max(0, int(self.get_parameter("rule_bootstrap_goal_count").value))
        self.class_weights = parse_class_weights(str(self.get_parameter("class_weights_json").value))
        self.avoid_distance = float(self.get_parameter("avoid_distance").value)
        self.free_gain_scale = float(self.get_parameter("free_gain_scale").value)
        self.rule_w1 = float(self.get_parameter("rule_w1").value)
        self.rule_w2 = float(self.get_parameter("rule_w2").value)
        self.rule_w3 = float(self.get_parameter("rule_w3").value)
        self.rule_w4 = float(self.get_parameter("rule_w4").value)
        self.rule_w5 = float(self.get_parameter("rule_w5").value)

        self.semantic_grid: OccupancyGrid | None = None
        self.latest_objects: list[dict[str, Any]] = []
        self.pose_x = 0.0
        self.pose_y = 0.0
        self.yaw = 0.0
        self.odom_ok = False
        self.front_range = math.inf

        self.last_goal_sec = 0.0
        self.last_goal_x = float("nan")
        self.last_goal_y = float("nan")
        self.goals_published = 0
        self.start_sec = self._now_sec()
        self._last_scoring_mode = ""

        self.policy_weights: dict[str, float] = {}
        self.policy_bias = 0.0
        self.policy_loaded = False
        self.policy_mtime: float | None = None
        self._policy_missing_warned = False
        self._last_no_goal_warn_sec = 0.0
        self._load_policy(force=True)

        self.selected_goal_pub = self.create_publisher(PoseStamped, selected_goal_topic, 10)
        self.scores_pub = self.create_publisher(String, scores_topic, 10)

        self.create_subscription(String, candidates_topic, self.on_candidates, 10)
        self.create_subscription(OccupancyGrid, semantic_grid_topic, self.on_semantic_grid, 10)
        self.create_subscription(String, semantic_obs_topic, self.on_semantic_observations, 10)
        self.create_subscription(Odometry, odom_topic, self.on_odom, 10)
        self.create_subscription(LaserScan, scan_topic, self.on_scan, 10)
        self.add_on_set_parameters_callback(self._on_set_parameters)
        self.create_service(Trigger, "/semantic_rl/reload_policy", self.on_reload_policy)
        self.create_timer(1.0, self._load_policy)

        self.get_logger().info(
            "semantic_rl_decider started: "
            f"candidates={candidates_topic}, policy={self.policy_file}, "
            f"score_mode={self.score_mode}, rule_bootstrap_sec={self.rule_bootstrap_sec:.1f}, "
            f"rule_bootstrap_goal_count={self.rule_bootstrap_goal_count}"
        )

    def on_semantic_grid(self, msg: OccupancyGrid) -> None:
        self.semantic_grid = msg

    def on_semantic_observations(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        objs = payload.get("objects", [])
        if isinstance(objs, list):
            self.latest_objects = objs[-400:]

    def on_odom(self, msg: Odometry) -> None:
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        self.pose_x = float(p.x)
        self.pose_y = float(p.y)
        self.yaw = yaw_from_quaternion(float(q.x), float(q.y), float(q.z), float(q.w))
        self.odom_ok = True

    def on_scan(self, msg: LaserScan) -> None:
        front = math.inf
        for i, r in enumerate(msg.ranges):
            if not math.isfinite(r):
                continue
            if r < msg.range_min or r > msg.range_max:
                continue
            angle = msg.angle_min + i * msg.angle_increment
            if abs(angle) <= 0.35:
                front = min(front, r)
        self.front_range = front

    def on_candidates(self, msg: String) -> None:
        if not self.odom_ok:
            return

        try:
            payload = json.loads(msg.data)
        except json.JSONDecodeError:
            self.get_logger().warn("Invalid candidates JSON")
            return

        frame_id = str(payload.get("frame_id", "map"))
        candidates = payload.get("candidates", [])
        if not isinstance(candidates, list) or not candidates:
            return

        use_policy, mode_reason = self._select_scoring_mode()
        mode_name = "policy" if use_policy else "rule"
        if mode_name != self._last_scoring_mode:
            self.get_logger().info(f"Scoring mode switched to '{mode_name}' ({mode_reason})")
            self._last_scoring_mode = mode_name

        scored: list[dict[str, Any]] = []
        for cand in candidates:
            x = float(cand.get("x", 0.0))
            y = float(cand.get("y", 0.0))
            free_gain_raw = float(cand.get("free_gain", 0.0))

            dist = math.hypot(x - self.pose_x, y - self.pose_y)
            heading = heading_delta(self.yaw, x, y, self.pose_x, self.pose_y)
            risk = clamp((self.avoid_distance - self.front_range) / max(self.avoid_distance, 1e-6), 0.0, 1.0)
            novelty = self._semantic_novelty(x, y)
            priority = priority_from_objects(x, y, self.latest_objects, self.class_weights)
            features = {
                "distance_to_frontier": dist,
                "estimated_free_gain": clamp(free_gain_raw / max(self.free_gain_scale, 1e-6), 0.0, 3.0),
                "heading_change": heading,
                "local_obstacle_risk": risk,
                "semantic_novelty_score": novelty,
                "semantic_priority_score": priority,
            }

            pseudo = rule_score(features, self.rule_w1, self.rule_w2, self.rule_w3, self.rule_w4, self.rule_w5)
            if use_policy:
                score = linear_policy_score(features, self.policy_weights, self.policy_bias)
            else:
                score = pseudo

            scored.append(
                {
                    "x": x,
                    "y": y,
                    "score": score,
                    "pseudo_expert_score": pseudo,
                    "features": features,
                }
            )

        scored.sort(key=lambda item: float(item["score"]), reverse=True)
        if not scored:
            return

        out = String()
        out.data = json.dumps(
            {
                "frame_id": frame_id,
                "stamp_sec": self._now_sec(),
                "policy_loaded": self.policy_loaded,
                "scoring_mode": mode_name,
                "scoring_reason": mode_reason,
                "candidates": scored,
            }
        )
        self.scores_pub.publish(out)

        best = None
        for cand in scored:
            x = float(cand["x"])
            y = float(cand["y"])
            if math.hypot(x - self.pose_x, y - self.pose_y) < self.min_robot_goal_distance:
                continue
            if not self._should_publish_goal(x, y):
                continue
            best = cand
            break

        # Deadlock breaker:
        # if every candidate is "too close", still allow a small move goal above a hard floor.
        if best is None and self.allow_close_goal_fallback:
            for cand in scored:
                x = float(cand["x"])
                y = float(cand["y"])
                dist = math.hypot(x - self.pose_x, y - self.pose_y)
                if dist < self.close_goal_min_distance:
                    continue
                # In close-goal fallback mode we intentionally allow re-publishing
                # the same nearby candidate after cooldown to avoid deadlock when
                # there is only one frontier candidate near the robot.
                if not self._cooldown_elapsed():
                    continue
                best = cand
                self.get_logger().info(
                    "Using close-goal fallback "
                    f"(dist={dist:.3f}, min_robot_goal_distance={self.min_robot_goal_distance:.3f}, "
                    f"close_goal_min_distance={self.close_goal_min_distance:.3f})"
                )
                break

        # If candidates are repeatedly filtered only because they are close to
        # the last goal, allow reissuing after cooldown to avoid getting stuck.
        if best is None and self.allow_duplicate_goal_fallback:
            for cand in scored:
                x = float(cand["x"])
                y = float(cand["y"])
                if math.hypot(x - self.pose_x, y - self.pose_y) < self.min_robot_goal_distance:
                    continue
                if not self._cooldown_elapsed():
                    continue
                best = cand
                self.get_logger().info(
                    "Using duplicate-goal fallback "
                    f"(dist_to_robot={math.hypot(x - self.pose_x, y - self.pose_y):.3f}, "
                    f"min_goal_distance={self.min_goal_distance:.3f})"
                )
                break
        if best is None:
            now = self._now_sec()
            if (now - self._last_no_goal_warn_sec) > 5.0:
                nearest = min(math.hypot(float(c["x"]) - self.pose_x, float(c["y"]) - self.pose_y) for c in scored)
                self.get_logger().warn(
                    "No goal published: all candidates filtered "
                    f"(count={len(scored)}, nearest_dist={nearest:.3f}, "
                    f"min_robot_goal_distance={self.min_robot_goal_distance:.3f})"
                )
                self._last_no_goal_warn_sec = now
            return

        goal = PoseStamped()
        goal.header.frame_id = frame_id
        # Use zero stamp so Nav2 transforms with latest available TF and avoids
        # occasional small future-extrapolation failures in simulation.
        goal.header.stamp = Time()
        goal.pose.position.x = float(best["x"])
        goal.pose.position.y = float(best["y"])
        goal.pose.orientation.w = 1.0
        self.selected_goal_pub.publish(goal)

        self.last_goal_sec = self._now_sec()
        self.last_goal_x = float(best["x"])
        self.last_goal_y = float(best["y"])
        self.goals_published += 1

    def _semantic_novelty(self, x: float, y: float) -> float:
        if self.semantic_grid is None:
            return 1.0

        info = self.semantic_grid.info
        width = int(info.width)
        height = int(info.height)
        center = map_index(
            x,
            y,
            float(info.resolution),
            width,
            height,
            float(info.origin.position.x),
            float(info.origin.position.y),
        )
        if center is None:
            return 1.0

        cx = center % width
        cy = center // width
        vals: list[int] = []
        for yy in range(max(0, cy - 2), min(height, cy + 3)):
            for xx in range(max(0, cx - 2), min(width, cx + 3)):
                vals.append(int(self.semantic_grid.data[yy * width + xx]))
        return novelty_from_semantic_grid(vals)

    def _should_publish_goal(self, x: float, y: float) -> bool:
        if not self._cooldown_elapsed():
            return False
        if math.isfinite(self.last_goal_x) and math.hypot(x - self.last_goal_x, y - self.last_goal_y) < self.min_goal_distance:
            return False
        return True

    def _cooldown_elapsed(self) -> bool:
        now = self._now_sec()
        return (now - self.last_goal_sec) >= self.goal_cooldown_sec

    def _on_set_parameters(self, params) -> SetParametersResult:
        for param in params:
            if param.name == "score_mode":
                value = str(param.value).strip().lower()
                if value not in ("auto", "rule_only", "policy_only"):
                    return SetParametersResult(successful=False, reason="score_mode must be auto|rule_only|policy_only")
                self.score_mode = value
                # Force one info log on next scoring cycle.
                self._last_scoring_mode = ""
            elif param.name == "rule_bootstrap_sec":
                self.rule_bootstrap_sec = max(0.0, float(param.value))
            elif param.name == "rule_bootstrap_goal_count":
                self.rule_bootstrap_goal_count = max(0, int(param.value))
        return SetParametersResult(successful=True)

    def _select_scoring_mode(self) -> tuple[bool, str]:
        if self.score_mode == "rule_only":
            return False, "score_mode=rule_only"

        if self.score_mode == "policy_only":
            if self.policy_loaded:
                return True, "score_mode=policy_only"
            return False, "policy_missing_policy_only_fallback_rule"

        # auto mode: rule-first warmup, then RL policy when ready.
        if not self.policy_loaded:
            return False, "policy_not_loaded"

        elapsed = self._now_sec() - self.start_sec
        if elapsed < self.rule_bootstrap_sec:
            return False, f"bootstrap_time<{self.rule_bootstrap_sec:.1f}s"
        if self.goals_published < self.rule_bootstrap_goal_count:
            return False, f"bootstrap_goals<{self.rule_bootstrap_goal_count}"
        return True, "bootstrap_complete"

    def _load_policy(self, force: bool = False) -> None:
        path = Path(self.policy_file)
        if not path.exists():
            self.policy_loaded = False
            if not self._policy_missing_warned:
                self.get_logger().warn(f"Policy file not found, using rule scoring fallback: {path}")
                self._policy_missing_warned = True
            return

        mtime = path.stat().st_mtime
        if not force and self.policy_mtime is not None and mtime <= self.policy_mtime:
            return

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            self.get_logger().warn(f"Failed to load semantic policy: {exc}")
            self.policy_loaded = False
            return

        weights = data.get("weights", {})
        if not isinstance(weights, dict):
            self.get_logger().warn("Invalid policy format (weights missing). Using rule scoring fallback.")
            self.policy_loaded = False
            return

        parsed_weights = {k: float(v) for k, v in weights.items() if k in FEATURE_ORDER}
        if not parsed_weights:
            self.get_logger().warn(
                f"Policy has no supported feature weights. Using rule scoring fallback: {path}"
            )
            self.policy_loaded = False
            self.policy_weights = {}
            self.policy_bias = 0.0
            self.policy_mtime = mtime
            return

        self.policy_weights = parsed_weights
        self.policy_bias = float(data.get("bias", 0.0))
        self.policy_mtime = mtime
        self.policy_loaded = True
        self._policy_missing_warned = False
        self.get_logger().info(f"Loaded semantic policy from {path}")

    def on_reload_policy(self, _req: Trigger.Request, resp: Trigger.Response) -> Trigger.Response:
        self._load_policy(force=True)
        resp.success = self.policy_loaded
        resp.message = "loaded" if self.policy_loaded else "fallback_rule_mode"
        return resp

    def _now_sec(self) -> float:
        return self.get_clock().now().nanoseconds / 1e9


def main() -> None:
    rclpy.init()
    node = SemanticRLDeciderNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
