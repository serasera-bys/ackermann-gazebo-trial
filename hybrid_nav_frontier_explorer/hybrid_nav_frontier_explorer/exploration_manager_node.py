from __future__ import annotations

import json
import math
from dataclasses import dataclass

import rclpy
from builtin_interfaces.msg import Time
from geometry_msgs.msg import PoseArray, PoseStamped, Twist
from nav_msgs.msg import Odometry
from nav2_msgs.action import NavigateToPose
from nav2_msgs.srv import ManageLifecycleNodes
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
    last_goal_distance: float = float("inf")


class ExplorationManagerNode(Node):
    def __init__(self) -> None:
        super().__init__("exploration_manager")

        self.declare_parameter("selected_goal_topic", "/semantic_rl/selected_goal")
        self.declare_parameter("frontier_candidates_topic", "/exploration/frontier_candidates")
        self.declare_parameter("status_topic", "/exploration/status")
        self.declare_parameter("odom_topic", "/ackermann_steering_controller/odometry")
        self.declare_parameter("cmd_vel_topic", "/cmd_vel")
        self.declare_parameter("goal_timeout_sec", 45.0)
        self.declare_parameter("goal_reissue_cooldown_sec", 2.0)
        self.declare_parameter("use_frontier_fallback", False)
        self.declare_parameter("enable_adaptive_frontier_fallback", True)
        self.declare_parameter("failure_streak_forced_fallback", 2)
        self.declare_parameter("forced_fallback_duration_sec", 20.0)
        self.declare_parameter("fallback_goal_min_separation", 0.5)
        self.declare_parameter("max_fallback_goal_distance", 6.0)
        self.declare_parameter("enable_stuck_cancel", True)
        self.declare_parameter("stuck_no_progress_timeout_sec", 25.0)
        self.declare_parameter("stuck_no_progress_timeout_recovery_sec", 35.0)
        self.declare_parameter("stuck_progress_distance", 0.03)
        self.declare_parameter("stuck_goal_progress_epsilon", 0.08)
        self.declare_parameter("stuck_rotation_angular_threshold", 0.35)
        self.declare_parameter("stuck_rotation_linear_max", 0.05)
        self.declare_parameter("failed_goal_blacklist_radius", 0.7)
        self.declare_parameter("failed_goal_blacklist_ttl_sec", 45.0)
        self.declare_parameter("nav_server_warn_period_sec", 5.0)
        self.declare_parameter("nav_auto_startup_if_unavailable", True)
        self.declare_parameter("nav_startup_retry_sec", 15.0)

        selected_goal_topic = str(self.get_parameter("selected_goal_topic").value)
        frontier_topic = str(self.get_parameter("frontier_candidates_topic").value)
        status_topic = str(self.get_parameter("status_topic").value)
        odom_topic = str(self.get_parameter("odom_topic").value)
        cmd_vel_topic = str(self.get_parameter("cmd_vel_topic").value)
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
        self.stuck_no_progress_timeout_recovery_sec = float(
            self.get_parameter("stuck_no_progress_timeout_recovery_sec").value
        )
        self.stuck_progress_distance = float(self.get_parameter("stuck_progress_distance").value)
        self.stuck_goal_progress_epsilon = float(self.get_parameter("stuck_goal_progress_epsilon").value)
        self.stuck_rotation_angular_threshold = float(
            self.get_parameter("stuck_rotation_angular_threshold").value
        )
        self.stuck_rotation_linear_max = float(self.get_parameter("stuck_rotation_linear_max").value)
        self.failed_goal_blacklist_radius = max(
            0.0, float(self.get_parameter("failed_goal_blacklist_radius").value)
        )
        self.failed_goal_blacklist_ttl_sec = max(
            0.0, float(self.get_parameter("failed_goal_blacklist_ttl_sec").value)
        )
        self.nav_server_warn_period_sec = max(
            0.5, float(self.get_parameter("nav_server_warn_period_sec").value)
        )
        self.nav_auto_startup_if_unavailable = bool(
            self.get_parameter("nav_auto_startup_if_unavailable").value
        )
        self.nav_startup_retry_sec = max(5.0, float(self.get_parameter("nav_startup_retry_sec").value))
        if self.stuck_goal_progress_epsilon <= 0.0:
            self.stuck_goal_progress_epsilon = max(self.stuck_progress_distance, 1e-3)
        if self.stuck_rotation_angular_threshold < 0.0:
            self.stuck_rotation_angular_threshold = 0.0
        if self.stuck_rotation_linear_max < 0.0:
            self.stuck_rotation_linear_max = 0.0
        if self.stuck_no_progress_timeout_recovery_sec < self.stuck_no_progress_timeout_sec:
            self.stuck_no_progress_timeout_recovery_sec = self.stuck_no_progress_timeout_sec

        self.goal_state = GoalState()
        self.last_goal_finished_sec = 0.0
        self.active_goal_handle = None
        self.failure_streak = 0
        self.force_frontier_until_sec = 0.0
        self.last_failed_goal_x = float("nan")
        self.last_failed_goal_y = float("nan")
        self.failed_goal_blacklist: list[tuple[float, float, float]] = []
        self.nav_server_ready = False
        self.last_nav_server_warn_sec = 0.0
        self.nav_server_wait_started_sec = 0.0
        self.last_nav_startup_attempt_sec = 0.0
        self.last_nav_startup_service_warn_sec = 0.0
        self.nav_startup_inflight = False
        self.node_started_sec = self._now_sec()
        self.any_goal_accepted = False

        self.latest_frontiers: PoseArray | None = None
        self.pending_goal: PoseStamped | None = None
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.odom_ok = False
        self.latest_cmd_linear = 0.0
        self.latest_cmd_angular = 0.0
        self.latest_cmd_sec = 0.0

        self.nav_client = ActionClient(self, NavigateToPose, "navigate_to_pose")
        self.lifecycle_client = self.create_client(
            ManageLifecycleNodes, "/lifecycle_manager_navigation/manage_nodes"
        )
        self.status_pub = self.create_publisher(String, status_topic, 10)

        self.create_subscription(PoseStamped, selected_goal_topic, self.on_selected_goal, 10)
        self.create_subscription(PoseArray, frontier_topic, self.on_frontiers, 10)
        self.create_subscription(Odometry, odom_topic, self.on_odom, 10)
        self.create_subscription(Twist, cmd_vel_topic, self.on_cmd_vel, 10)
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

    def on_cmd_vel(self, msg: Twist) -> None:
        self.latest_cmd_linear = float(msg.linear.x)
        self.latest_cmd_angular = float(msg.angular.z)
        self.latest_cmd_sec = self._now_sec()

    def on_timer(self) -> None:
        now = self._now_sec()
        self._prune_blacklist(now)

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
            goal_dist = math.hypot(
                self.goal_state.target_x - self.robot_x,
                self.goal_state.target_y - self.robot_y,
            )
            # Movement jitter can exceed a few centimeters while still making no
            # real progress to the target. Use distance-to-goal improvement as
            # the stuck criterion to avoid waiting forever in oscillation loops.
            if (self.goal_state.last_goal_distance - goal_dist) >= self.stuck_goal_progress_epsilon:
                self.goal_state.last_goal_distance = goal_dist
                self.goal_state.last_progress_sec = now
            elif goal_dist <= 0.25:
                # Near-goal behavior can be oscillatory; avoid premature cancel.
                self.goal_state.last_progress_sec = now
            elif self._is_rotation_recovery_active(now):
                # Let Nav2 recovery spins complete before declaring stuck.
                self.goal_state.last_progress_sec = now
            else:
                no_progress_timeout = self.stuck_no_progress_timeout_sec
                if self._is_low_speed_recovery_active(now):
                    no_progress_timeout = self.stuck_no_progress_timeout_recovery_sec
                if (now - self.goal_state.last_progress_sec) <= no_progress_timeout:
                    return
                self.get_logger().warn(
                    "No progress detected. Cancelling current navigation goal "
                    f"(timeout={no_progress_timeout:.1f}s, "
                    f"goal_dist={goal_dist:.2f}m, progress_eps={self.stuck_goal_progress_epsilon:.2f}m)."
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

        # Action endpoints may exist even when Nav2 lifecycle nodes are still
        # unconfigured/inactive. Keep requesting STARTUP during bootstrapping
        # until at least one goal is accepted.
        if not self.any_goal_accepted:
            self._try_nav_startup(now, now - self.node_started_sec)

        if self.nav_server_ready is False:
            if self.nav_client.wait_for_server(timeout_sec=0.2):
                self.nav_server_ready = True
                self.nav_server_wait_started_sec = 0.0
                self.get_logger().info("navigate_to_pose action server is now available")
            else:
                if self.nav_server_wait_started_sec <= 0.0:
                    self.nav_server_wait_started_sec = now
                waited = now - self.nav_server_wait_started_sec
                if (now - self.last_nav_server_warn_sec) >= self.nav_server_warn_period_sec:
                    if waited >= 30.0:
                        self.get_logger().warn(
                            "navigate_to_pose action server not available yet "
                            f"(waited={waited:.1f}s). Check bt_navigator/lifecycle_manager_navigation status."
                        )
                    else:
                        self.get_logger().warn("navigate_to_pose action server not available yet")
                    self.last_nav_server_warn_sec = now
                self._try_nav_startup(now, waited)
                return

        # Always send zero-stamped goals so Nav2 transforms against the latest TF.
        # This avoids intermittent future/past extrapolation around simulation clock jitter.
        goal_for_nav = PoseStamped()
        goal_for_nav.header.frame_id = goal.header.frame_id
        goal_for_nav.header.stamp = Time()
        goal_for_nav.pose = goal.pose

        nav_goal = NavigateToPose.Goal()
        nav_goal.pose = goal_for_nav
        send_future = self.nav_client.send_goal_async(nav_goal)
        send_future.add_done_callback(self._on_goal_response)

        self.goal_state.active = True
        self.goal_state.cancel_requested = False
        self.goal_state.timed_out = False
        self.goal_state.stuck_out = False
        self.goal_state.started_sec = now
        self.goal_state.target_x = float(goal_for_nav.pose.position.x)
        self.goal_state.target_y = float(goal_for_nav.pose.position.y)
        self.goal_state.source = goal_source
        self.goal_state.last_progress_sec = now
        self.goal_state.last_goal_distance = math.hypot(
            self.goal_state.target_x - self.robot_x,
            self.goal_state.target_y - self.robot_y,
        ) if self.odom_ok else float("inf")
        self.pending_goal = None
        self._publish_status("goal_sent", self.goal_state.target_x, self.goal_state.target_y, self.goal_state.source)

    def _on_goal_response(self, future) -> None:
        goal_handle = future.result()
        if goal_handle is None or not goal_handle.accepted:
            self.get_logger().warn("Navigation goal rejected")
            self.nav_server_ready = False
            self._try_nav_startup(self._now_sec(), self.nav_startup_retry_sec + 1.0)
            self.goal_state.active = False
            self.goal_state.cancel_requested = False
            self.goal_state.timed_out = False
            self.goal_state.stuck_out = False
            self.last_goal_finished_sec = self._now_sec()
            self._publish_status("goal_rejected", self.goal_state.target_x, self.goal_state.target_y, self.goal_state.source)
            self._mark_failure("goal_rejected")
            return

        self.active_goal_handle = goal_handle
        self.any_goal_accepted = True
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

        filtered: list[tuple[float, float, object]] = []
        for candidate in self.latest_frontiers.poses:
            cx = float(candidate.position.x)
            cy = float(candidate.position.y)
            if self._is_blacklisted(cx, cy):
                continue

            if self.odom_ok:
                dist_robot = math.hypot(cx - self.robot_x, cy - self.robot_y)
                if dist_robot > self.max_fallback_goal_distance:
                    continue
            else:
                dist_robot = 0.0

            dist_failed = 0.0
            if math.isfinite(self.last_failed_goal_x) and math.isfinite(self.last_failed_goal_y):
                dist_failed = math.hypot(cx - self.last_failed_goal_x, cy - self.last_failed_goal_y)
                if prefer_diverse and dist_failed < self.fallback_goal_min_separation:
                    continue

            filtered.append((dist_robot, dist_failed, candidate))

        if filtered:
            if prefer_diverse:
                diverse = [item for item in filtered if item[1] >= self.fallback_goal_min_separation]
                chosen = diverse if diverse else filtered
                # Prefer spatially valid alternatives but keep nearest-first to
                # avoid ping-ponging between two far frontier points.
                chosen.sort(key=lambda item: item[0])
                pose = chosen[0][2]
            else:
                filtered.sort(key=lambda item: item[0])
                pose = filtered[0][2]
        else:
            self.get_logger().warn("No valid frontier fallback goal available after blacklist/distance filtering.")
            return None

        out = PoseStamped()
        out.header.frame_id = self.latest_frontiers.header.frame_id
        out.header.stamp = Time()
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
        self._add_failed_goal_blacklist(self.goal_state.target_x, self.goal_state.target_y)

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

    def _is_rotation_recovery_active(self, now: float) -> bool:
        if (now - self.latest_cmd_sec) > 1.0:
            return False
        return (
            abs(self.latest_cmd_angular) >= self.stuck_rotation_angular_threshold
            and abs(self.latest_cmd_linear) <= self.stuck_rotation_linear_max
        )

    def _is_low_speed_recovery_active(self, now: float) -> bool:
        if (now - self.latest_cmd_sec) > 1.0:
            return False
        magnitude = abs(self.latest_cmd_linear) + abs(self.latest_cmd_angular)
        return (
            magnitude >= 0.02
            and
            abs(self.latest_cmd_linear) <= max(self.stuck_rotation_linear_max, 0.12)
            and abs(self.latest_cmd_angular) <= max(self.stuck_rotation_angular_threshold, 0.2)
        )

    def _add_failed_goal_blacklist(self, x: float, y: float) -> None:
        if self.failed_goal_blacklist_ttl_sec <= 0.0:
            return
        now = self._now_sec()
        self.failed_goal_blacklist.append((x, y, now + self.failed_goal_blacklist_ttl_sec))

    def _prune_blacklist(self, now: float) -> None:
        if not self.failed_goal_blacklist:
            return
        self.failed_goal_blacklist = [v for v in self.failed_goal_blacklist if v[2] > now]

    def _is_blacklisted(self, x: float, y: float) -> bool:
        if not self.failed_goal_blacklist or self.failed_goal_blacklist_radius <= 0.0:
            return False
        for bx, by, _ in self.failed_goal_blacklist:
            if math.hypot(x - bx, y - by) <= self.failed_goal_blacklist_radius:
                return True
        return False

    def _try_nav_startup(self, now: float, waited: float) -> None:
        if not self.nav_auto_startup_if_unavailable:
            return
        if waited < self.nav_startup_retry_sec:
            return
        if self.nav_startup_inflight:
            return
        if (now - self.last_nav_startup_attempt_sec) < self.nav_startup_retry_sec:
            return
        if not self.lifecycle_client.wait_for_service(timeout_sec=0.2):
            if (now - self.last_nav_startup_service_warn_sec) >= self.nav_server_warn_period_sec:
                self.get_logger().warn(
                    "lifecycle_manager_navigation/manage_nodes service unavailable; "
                    "cannot request Nav2 STARTUP yet"
                )
                self.last_nav_startup_service_warn_sec = now
            return

        req = ManageLifecycleNodes.Request()
        req.command = ManageLifecycleNodes.Request.STARTUP
        future = self.lifecycle_client.call_async(req)
        future.add_done_callback(self._on_nav_startup_response)
        self.nav_startup_inflight = True
        self.last_nav_startup_attempt_sec = now
        self.get_logger().warn("Requested Nav2 lifecycle STARTUP (action server unavailable)")

    def _on_nav_startup_response(self, future) -> None:
        self.nav_startup_inflight = False
        try:
            resp = future.result()
            if bool(resp.success):
                self.get_logger().info("Nav2 lifecycle STARTUP request succeeded")
            else:
                self.get_logger().warn("Nav2 lifecycle STARTUP request failed")
        except Exception as exc:
            self.get_logger().warn(f"Nav2 lifecycle STARTUP call failed: {exc}")


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
