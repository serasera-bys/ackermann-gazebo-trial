from __future__ import annotations

import json
import math
from dataclasses import dataclass

import rclpy
from geometry_msgs.msg import PoseArray, PoseStamped
from nav_msgs.msg import Odometry
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
from rclpy.node import Node
from std_msgs.msg import String


@dataclass(slots=True)
class GoalState:
    active: bool = False
    cancel_requested: bool = False
    timed_out: bool = False
    stuck_out: bool = False
    started_sec: float = 0.0
    target_x: float = 0.0
    target_y: float = 0.0
    source: str = "rl"
    last_progress_sec: float = 0.0
    last_progress_x: float = 0.0
    last_progress_y: float = 0.0


class ExplorationManagerNode(Node):
    def __init__(self) -> None:
        super().__init__("exploration_manager")

        self.declare_parameter("selected_goal_topic", "/semantic_rl/selected_goal")
        self.declare_parameter("frontier_candidates_topic", "/exploration/frontier_candidates")
        self.declare_parameter("status_topic", "/exploration/status")
        self.declare_parameter("odom_topic", "/ackermann_steering_controller/odometry")
        self.declare_parameter("goal_timeout_sec", 45.0)
        self.declare_parameter("goal_reissue_cooldown_sec", 2.0)
        self.declare_parameter("use_frontier_fallback", False)
        self.declare_parameter("enable_adaptive_frontier_fallback", True)
        self.declare_parameter("failure_streak_forced_fallback", 2)
        self.declare_parameter("forced_fallback_duration_sec", 20.0)
        self.declare_parameter("fallback_goal_min_separation", 0.5)
        self.declare_parameter("max_fallback_goal_distance", 4.0)
        self.declare_parameter("enable_stuck_cancel", True)
        self.declare_parameter("stuck_no_progress_timeout_sec", 12.0)
        self.declare_parameter("stuck_progress_distance", 0.08)

        selected_goal_topic = str(self.get_parameter("selected_goal_topic").value)
        frontier_topic = str(self.get_parameter("frontier_candidates_topic").value)
        status_topic = str(self.get_parameter("status_topic").value)
        odom_topic = str(self.get_parameter("odom_topic").value)
        self.goal_timeout_sec = float(self.get_parameter("goal_timeout_sec").value)
        self.goal_reissue_cooldown_sec = float(self.get_parameter("goal_reissue_cooldown_sec").value)
        self.use_frontier_fallback = bool(self.get_parameter("use_frontier_fallback").value)
        self.enable_adaptive_frontier_fallback = bool(
            self.get_parameter("enable_adaptive_frontier_fallback").value
        )
        self.failure_streak_forced_fallback = int(self.get_parameter("failure_streak_forced_fallback").value)
        self.forced_fallback_duration_sec = float(self.get_parameter("forced_fallback_duration_sec").value)
        self.fallback_goal_min_separation = float(self.get_parameter("fallback_goal_min_separation").value)
        self.max_fallback_goal_distance = float(self.get_parameter("max_fallback_goal_distance").value)
        self.enable_stuck_cancel = bool(self.get_parameter("enable_stuck_cancel").value)
        self.stuck_no_progress_timeout_sec = float(self.get_parameter("stuck_no_progress_timeout_sec").value)
        self.stuck_progress_distance = float(self.get_parameter("stuck_progress_distance").value)

        self.goal_state = GoalState()
        self.last_goal_finished_sec = 0.0
        self.active_goal_handle = None
        self.failure_streak = 0
        self.force_frontier_until_sec = 0.0
        self.last_failed_goal_x = float("nan")
        self.last_failed_goal_y = float("nan")

        self.latest_frontiers: PoseArray | None = None
        self.pending_goal: PoseStamped | None = None
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.odom_ok = False

        self.nav_client = ActionClient(self, NavigateToPose, "navigate_to_pose")
        self.status_pub = self.create_publisher(String, status_topic, 10)

        self.create_subscription(PoseStamped, selected_goal_topic, self.on_selected_goal, 10)
        self.create_subscription(PoseArray, frontier_topic, self.on_frontiers, 10)
        self.create_subscription(Odometry, odom_topic, self.on_odom, 10)
        self.create_timer(0.5, self.on_timer)

        self.get_logger().info(
            f"exploration_manager started: selected_goal={selected_goal_topic}, frontiers={frontier_topic}"
        )

    def on_selected_goal(self, msg: PoseStamped) -> None:
        self.pending_goal = msg

    def on_frontiers(self, msg: PoseArray) -> None:
        self.latest_frontiers = msg

    def on_odom(self, msg: Odometry) -> None:
        self.robot_x = float(msg.pose.pose.position.x)
        self.robot_y = float(msg.pose.pose.position.y)
        self.odom_ok = True

    def on_timer(self) -> None:
        now = self._now_sec()

        if (
            self.goal_state.active
            and not self.goal_state.cancel_requested
            and (now - self.goal_state.started_sec) > self.goal_timeout_sec
        ):
            self.get_logger().warn("Goal timeout. Cancelling current navigation goal.")
            if self.active_goal_handle is not None:
                self.active_goal_handle.cancel_goal_async()
            self.goal_state.cancel_requested = True
            self.goal_state.timed_out = True
            self._publish_status("timeout", self.goal_state.target_x, self.goal_state.target_y, self.goal_state.source)

        if (
            self.goal_state.active
            and self.enable_stuck_cancel
            and self.odom_ok
            and not self.goal_state.cancel_requested
        ):
            moved = math.hypot(
                self.robot_x - self.goal_state.last_progress_x,
                self.robot_y - self.goal_state.last_progress_y,
            )
            if moved >= self.stuck_progress_distance:
                self.goal_state.last_progress_x = self.robot_x
                self.goal_state.last_progress_y = self.robot_y
                self.goal_state.last_progress_sec = now
            elif (now - self.goal_state.last_progress_sec) > self.stuck_no_progress_timeout_sec:
                self.get_logger().warn(
                    "No progress detected. Cancelling current navigation goal "
                    f"(timeout={self.stuck_no_progress_timeout_sec:.1f}s, moved<{self.stuck_progress_distance:.2f}m)."
                )
                if self.active_goal_handle is not None:
                    self.active_goal_handle.cancel_goal_async()
                self.goal_state.cancel_requested = True
                self.goal_state.stuck_out = True
                self._publish_status("stuck", self.goal_state.target_x, self.goal_state.target_y, self.goal_state.source)

        if self.goal_state.active:
            return

        if (now - self.last_goal_finished_sec) < self.goal_reissue_cooldown_sec:
            return

        force_frontier = self.use_frontier_fallback and now < self.force_frontier_until_sec
        goal_source = "rl"

        if force_frontier:
            goal = self._fallback_goal_from_frontiers(prefer_diverse=True)
            goal_source = "frontier_forced"
            if goal is None:
                goal = self.pending_goal
                goal_source = "rl"
            if goal is None:
                return
        else:
            goal = self.pending_goal
            if goal is None and self.use_frontier_fallback:
                goal = self._fallback_goal_from_frontiers(prefer_diverse=False)
                goal_source = "frontier"
                if goal is None:
                    return
            elif goal is None:
                return

        if self.nav_client.wait_for_server(timeout_sec=0.2) is False:
            self.get_logger().warn("navigate_to_pose action server not available yet")
            return

        nav_goal = NavigateToPose.Goal()
        nav_goal.pose = goal
        send_future = self.nav_client.send_goal_async(nav_goal)
        send_future.add_done_callback(self._on_goal_response)

        self.goal_state.active = True
        self.goal_state.cancel_requested = False
        self.goal_state.timed_out = False
        self.goal_state.stuck_out = False
        self.goal_state.started_sec = now
        self.goal_state.target_x = float(goal.pose.position.x)
        self.goal_state.target_y = float(goal.pose.position.y)
        self.goal_state.source = goal_source
        self.goal_state.last_progress_sec = now
        self.goal_state.last_progress_x = self.robot_x
        self.goal_state.last_progress_y = self.robot_y
        self.pending_goal = None
        self._publish_status("goal_sent", self.goal_state.target_x, self.goal_state.target_y, self.goal_state.source)

    def _on_goal_response(self, future) -> None:
        goal_handle = future.result()
        if goal_handle is None or not goal_handle.accepted:
            self.get_logger().warn("Navigation goal rejected")
            self.goal_state.active = False
            self.goal_state.cancel_requested = False
            self.goal_state.timed_out = False
            self.goal_state.stuck_out = False
            self.last_goal_finished_sec = self._now_sec()
            self._publish_status("goal_rejected", self.goal_state.target_x, self.goal_state.target_y, self.goal_state.source)
            self._mark_failure("goal_rejected")
            return

        self.active_goal_handle = goal_handle
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._on_goal_result)

    def _on_goal_result(self, future) -> None:
        status_text = "goal_unknown"
        try:
            result = future.result()
            status = int(result.status)
            if status == 4:
                status_text = "goal_succeeded"
            elif status == 6:
                status_text = "goal_aborted"
            elif status == 5:
                status_text = "goal_canceled"
            else:
                status_text = f"goal_status_{status}"
        except Exception:
            status_text = "goal_result_error"

        timed_out = self.goal_state.timed_out
        stuck_out = self.goal_state.stuck_out
        self.goal_state.active = False
        self.goal_state.cancel_requested = False
        self.goal_state.timed_out = False
        self.goal_state.stuck_out = False
        self.active_goal_handle = None
        self.last_goal_finished_sec = self._now_sec()
        self._publish_status(status_text, self.goal_state.target_x, self.goal_state.target_y, self.goal_state.source)

        if status_text == "goal_succeeded":
            self.failure_streak = 0
            self.force_frontier_until_sec = 0.0
            return

        if timed_out and status_text == "goal_canceled":
            self._mark_failure("timeout")
            return
        if stuck_out and status_text == "goal_canceled":
            self._mark_failure("stuck")
            return

        self._mark_failure(status_text)

    def _fallback_goal_from_frontiers(self, prefer_diverse: bool) -> PoseStamped | None:
        if self.latest_frontiers is None or not self.latest_frontiers.poses:
            return None

        filtered: list[tuple[float, object]] = []
        for candidate in self.latest_frontiers.poses:
            cx = float(candidate.position.x)
            cy = float(candidate.position.y)

            if self.odom_ok:
                dist_robot = math.hypot(cx - self.robot_x, cy - self.robot_y)
                if dist_robot > self.max_fallback_goal_distance:
                    continue
            else:
                dist_robot = 0.0

            if prefer_diverse and math.isfinite(self.last_failed_goal_x) and math.isfinite(self.last_failed_goal_y):
                dist_failed = math.hypot(cx - self.last_failed_goal_x, cy - self.last_failed_goal_y)
                if dist_failed < self.fallback_goal_min_separation:
                    continue

            filtered.append((dist_robot, candidate))

        if filtered:
            filtered.sort(key=lambda item: item[0])
            pose = filtered[0][1]
        else:
            # As a last resort, preserve old behavior (first frontier).
            pose = self.latest_frontiers.poses[0]

        out = PoseStamped()
        out.header = self.latest_frontiers.header
        out.pose = pose
        return out

    def _publish_status(self, event: str, x: float, y: float, source: str) -> None:
        msg = String()
        msg.data = json.dumps(
            {
                "event": event,
                "stamp_sec": self._now_sec(),
                "target_x": x,
                "target_y": y,
                "source": source,
            }
        )
        self.status_pub.publish(msg)

    def _mark_failure(self, event: str) -> None:
        self.failure_streak += 1
        self.last_failed_goal_x = self.goal_state.target_x
        self.last_failed_goal_y = self.goal_state.target_y

        if not self.use_frontier_fallback or not self.enable_adaptive_frontier_fallback:
            return

        if self.failure_streak < max(self.failure_streak_forced_fallback, 1):
            return

        now = self._now_sec()
        was_forced = now < self.force_frontier_until_sec
        self.force_frontier_until_sec = max(self.force_frontier_until_sec, now + self.forced_fallback_duration_sec)
        if not was_forced:
            self.get_logger().warn(
                "Adaptive fallback enabled after repeated failures "
                f"(event={event}, streak={self.failure_streak}, duration={self.forced_fallback_duration_sec:.1f}s)"
            )

    def _now_sec(self) -> float:
        return self.get_clock().now().nanoseconds / 1e9


def main() -> None:
    rclpy.init()
    node = ExplorationManagerNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
