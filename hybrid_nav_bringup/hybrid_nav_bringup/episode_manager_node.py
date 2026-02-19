import json
import math
import random
import re
import subprocess
import time
from pathlib import Path

import rclpy
from ament_index_python.packages import PackageNotFoundError, get_package_share_directory
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.node import Node
from std_msgs.msg import Bool, String


def _default_controller_params_file() -> str:
    try:
        share = Path(get_package_share_directory("ackermann_bringup"))
        return str(share / "config" / "ros2_controllers.yaml")
    except PackageNotFoundError:
        return "ackermann_bringup/config/ros2_controllers.yaml"


class EpisodeManagerNode(Node):
    def __init__(self) -> None:
        super().__init__("hybrid_episode_manager")

        self.declare_parameter("auto_reset_enabled", True)
        self.declare_parameter("reset_on_goal", True)
        self.declare_parameter("reset_on_stuck", True)
        self.declare_parameter("goal_x", 8.0)
        self.declare_parameter("goal_y", 0.0)
        self.declare_parameter("goal_tolerance", 0.25)
        self.declare_parameter("odom_topic", "/ackermann_steering_controller/odometry")
        self.declare_parameter("cmd_topic", "/cmd_vel")
        self.declare_parameter("safety_topic", "/safety_layer/intervention")
        self.declare_parameter("episode_event_topic", "/hybrid_nav/episode_event")
        self.declare_parameter("reset_pause_sec", 1.0)
        self.declare_parameter("world_control_service", "/world/default/control")
        self.declare_parameter("world_set_pose_service", "/world/default/set_pose")
        self.declare_parameter("reset_mode", "model_only")
        self.declare_parameter("progress_distance", 0.15)
        self.declare_parameter("goal_progress_distance", 0.12)
        self.declare_parameter("stuck_timeout_sec", 4.0)
        self.declare_parameter("min_episode_sec", 2.0)
        self.declare_parameter(
            "max_episode_sec",
            0.0,
            ParameterDescriptor(dynamic_typing=True),
        )
        self.declare_parameter("stuck_min_cmd_linear", 0.10)
        self.declare_parameter("cmd_stale_sec", 1.0)
        self.declare_parameter("post_reset_goal_guard_sec", 1.0)
        self.declare_parameter("post_reset_goal_rearm_distance", 0.8)
        self.declare_parameter("rearm_timeout_sec", 2.5)
        self.declare_parameter("fallback_all_reset_on_rearm_timeout", True)
        self.declare_parameter("respawn_controllers_after_all_reset", True)
        self.declare_parameter("controller_manager_name", "/controller_manager")
        self.declare_parameter(
            "controller_params_file",
            _default_controller_params_file(),
        )
        self.declare_parameter("controller_manager_timeout_sec", 120.0)
        self.declare_parameter("controller_service_call_timeout_sec", 30.0)
        self.declare_parameter("controller_respawn_retries", 3)
        self.declare_parameter("controller_respawn_retry_delay_sec", 0.8)
        self.declare_parameter("wait_for_controllers_active", True)
        self.declare_parameter("controllers_check_period_sec", 0.5)
        self.declare_parameter(
            "required_controllers",
            ["joint_state_broadcaster", "ackermann_steering_controller", "front_wheel_velocity_controller"],
        )
        self.declare_parameter("randomization_enabled", False)
        self.declare_parameter("random_seed", -1)
        self.declare_parameter("randomize_start_pose", True)
        self.declare_parameter("randomize_goal", True)
        self.declare_parameter("randomize_obstacles", True)
        self.declare_parameter("car_model_name", "ackermann_car")
        self.declare_parameter("obstacle_model_names", ["obstacle_box_1", "obstacle_box_2", "obstacle_box_3"])
        self.declare_parameter("start_x_min", -0.6)
        self.declare_parameter("start_x_max", 0.6)
        self.declare_parameter("start_y_min", -1.0)
        self.declare_parameter("start_y_max", 1.0)
        self.declare_parameter("start_yaw_min", -0.7)
        self.declare_parameter("start_yaw_max", 0.7)
        self.declare_parameter("goal_x_min", 4.8)
        self.declare_parameter("goal_x_max", 7.0)
        self.declare_parameter("goal_y_min", -1.2)
        self.declare_parameter("goal_y_max", 1.2)
        self.declare_parameter("obstacle_x_min", 1.3)
        self.declare_parameter("obstacle_x_max", 4.5)
        self.declare_parameter("obstacle_y_min", -1.2)
        self.declare_parameter("obstacle_y_max", 1.2)
        self.declare_parameter("randomization_min_clearance", 0.7)

        self.auto_reset_enabled = bool(self.get_parameter("auto_reset_enabled").value)
        self.reset_on_goal = bool(self.get_parameter("reset_on_goal").value)
        self.reset_on_stuck = bool(self.get_parameter("reset_on_stuck").value)
        self.goal_x = float(self.get_parameter("goal_x").value)
        self.goal_y = float(self.get_parameter("goal_y").value)
        self.goal_tolerance = float(self.get_parameter("goal_tolerance").value)
        self.odom_topic = str(self.get_parameter("odom_topic").value)
        self.cmd_topic = str(self.get_parameter("cmd_topic").value)
        self.safety_topic = str(self.get_parameter("safety_topic").value)
        self.episode_event_topic = str(self.get_parameter("episode_event_topic").value)
        self.reset_pause_sec = max(0.0, float(self.get_parameter("reset_pause_sec").value))
        self.world_control_service = str(self.get_parameter("world_control_service").value)
        self.world_set_pose_service = str(self.get_parameter("world_set_pose_service").value)
        self.reset_mode = str(self.get_parameter("reset_mode").value).strip().lower()
        if self.reset_mode not in ("all", "model_only", "time_only"):
            self.get_logger().warn(
                f"Unknown reset_mode='{self.reset_mode}', using 'model_only'"
            )
            self.reset_mode = "model_only"
        self.progress_distance = max(0.01, float(self.get_parameter("progress_distance").value))
        self.goal_progress_distance = max(
            0.0, float(self.get_parameter("goal_progress_distance").value)
        )
        self.stuck_timeout_sec = max(0.5, float(self.get_parameter("stuck_timeout_sec").value))
        self.min_episode_sec = max(0.0, float(self.get_parameter("min_episode_sec").value))
        self.max_episode_sec = max(0.0, float(self.get_parameter("max_episode_sec").value))
        self.stuck_min_cmd_linear = max(0.0, float(self.get_parameter("stuck_min_cmd_linear").value))
        self.cmd_stale_sec = max(0.1, float(self.get_parameter("cmd_stale_sec").value))
        self.post_reset_goal_guard_sec = max(
            0.0, float(self.get_parameter("post_reset_goal_guard_sec").value)
        )
        self.post_reset_goal_rearm_distance = max(
            0.0, float(self.get_parameter("post_reset_goal_rearm_distance").value)
        )
        self.rearm_timeout_sec = max(0.0, float(self.get_parameter("rearm_timeout_sec").value))
        self.fallback_all_reset_on_rearm_timeout = bool(
            self.get_parameter("fallback_all_reset_on_rearm_timeout").value
        )
        self.respawn_controllers_after_all_reset = bool(
            self.get_parameter("respawn_controllers_after_all_reset").value
        )
        self.controller_manager_name = str(self.get_parameter("controller_manager_name").value)
        self.controller_params_file = str(self.get_parameter("controller_params_file").value)
        self.controller_manager_timeout_sec = max(
            1.0, float(self.get_parameter("controller_manager_timeout_sec").value)
        )
        self.controller_service_call_timeout_sec = max(
            1.0, float(self.get_parameter("controller_service_call_timeout_sec").value)
        )
        self.controller_respawn_retries = max(
            1, int(self.get_parameter("controller_respawn_retries").value)
        )
        self.controller_respawn_retry_delay_sec = max(
            0.1, float(self.get_parameter("controller_respawn_retry_delay_sec").value)
        )
        self.wait_for_controllers_active = bool(
            self.get_parameter("wait_for_controllers_active").value
        )
        self.controllers_check_period_sec = max(
            0.1, float(self.get_parameter("controllers_check_period_sec").value)
        )
        self.required_controllers = [
            str(name).strip()
            for name in self.get_parameter("required_controllers").value
            if str(name).strip()
        ]
        self.randomization_enabled = bool(self.get_parameter("randomization_enabled").value)
        random_seed = int(self.get_parameter("random_seed").value)
        self.rng = random.Random()
        if random_seed >= 0:
            self.rng.seed(random_seed)
        self.randomize_start_pose = bool(self.get_parameter("randomize_start_pose").value)
        self.randomize_goal = bool(self.get_parameter("randomize_goal").value)
        self.randomize_obstacles = bool(self.get_parameter("randomize_obstacles").value)
        self.car_model_name = str(self.get_parameter("car_model_name").value)
        self.obstacle_model_names = [
            str(name).strip()
            for name in self.get_parameter("obstacle_model_names").value
            if str(name).strip()
        ]
        self.start_x_min = float(self.get_parameter("start_x_min").value)
        self.start_x_max = float(self.get_parameter("start_x_max").value)
        self.start_y_min = float(self.get_parameter("start_y_min").value)
        self.start_y_max = float(self.get_parameter("start_y_max").value)
        self.start_yaw_min = float(self.get_parameter("start_yaw_min").value)
        self.start_yaw_max = float(self.get_parameter("start_yaw_max").value)
        self.goal_x_min = float(self.get_parameter("goal_x_min").value)
        self.goal_x_max = float(self.get_parameter("goal_x_max").value)
        self.goal_y_min = float(self.get_parameter("goal_y_min").value)
        self.goal_y_max = float(self.get_parameter("goal_y_max").value)
        self.obstacle_x_min = float(self.get_parameter("obstacle_x_min").value)
        self.obstacle_x_max = float(self.get_parameter("obstacle_x_max").value)
        self.obstacle_y_min = float(self.get_parameter("obstacle_y_min").value)
        self.obstacle_y_max = float(self.get_parameter("obstacle_y_max").value)
        self.randomization_min_clearance = max(
            0.2, float(self.get_parameter("randomization_min_clearance").value)
        )
        self.start_x_min, self.start_x_max = self._ordered_range(self.start_x_min, self.start_x_max)
        self.start_y_min, self.start_y_max = self._ordered_range(self.start_y_min, self.start_y_max)
        self.start_yaw_min, self.start_yaw_max = self._ordered_range(
            self.start_yaw_min, self.start_yaw_max
        )
        self.goal_x_min, self.goal_x_max = self._ordered_range(self.goal_x_min, self.goal_x_max)
        self.goal_y_min, self.goal_y_max = self._ordered_range(self.goal_y_min, self.goal_y_max)
        self.obstacle_x_min, self.obstacle_x_max = self._ordered_range(
            self.obstacle_x_min, self.obstacle_x_max
        )
        self.obstacle_y_min, self.obstacle_y_max = self._ordered_range(
            self.obstacle_y_min, self.obstacle_y_max
        )

        self.pose_x = 0.0
        self.pose_y = 0.0
        self.odom_received = False
        self.odom_msg_count = 0
        self.latest_linear_cmd = 0.0
        self.last_cmd_time_sec = float("-inf")
        self.last_safety_hit_sec = float("-inf")

        self.episode_id = 0
        self.episode_active = False
        self.episode_start_time_sec = 0.0
        self.episode_start_odom_count = 0
        self.goal_guard_until_sec = 0.0
        self.last_progress_time_sec = 0.0
        self.progress_anchor_x = 0.0
        self.progress_anchor_y = 0.0
        self.best_goal_distance = math.inf
        self.goal_rearm_pending = False
        self.goal_rearm_started_sec = 0.0
        self._next_episode_after_reset = False
        self._randomization_applied_for_pending_reset = False

        self.pending_reset = False
        self.reset_ready_at_sec = 0.0
        self.reset_wait_started_sec = 0.0
        self.pending_reset_fallback_used = False
        self._last_controller_check_sec = float("-inf")
        self._last_controller_wait_log_sec = float("-inf")
        self._last_controller_check_ok = False
        self._last_controller_states: dict[str, str] = {}

        self.create_subscription(Odometry, self.odom_topic, self.on_odom, 20)
        self.create_subscription(Twist, self.cmd_topic, self.on_cmd, 20)
        self.create_subscription(Bool, self.safety_topic, self.on_safety, 20)
        self.event_pub = self.create_publisher(String, self.episode_event_topic, 10)
        self.create_timer(0.2, self.on_timer)

        self.get_logger().info(
            f"Episode manager started: auto_reset={self.auto_reset_enabled}, "
            f"goal=({self.goal_x:.2f}, {self.goal_y:.2f}), odom={self.odom_topic}, "
            f"event_topic={self.episode_event_topic}, randomization={self.randomization_enabled}"
        )

    def on_odom(self, msg: Odometry) -> None:
        self.pose_x = msg.pose.pose.position.x
        self.pose_y = msg.pose.pose.position.y
        self.odom_received = True
        self.odom_msg_count += 1

        if not self.episode_active and not self.pending_reset:
            if self.wait_for_controllers_active and not self._controllers_are_active(self._now_sec()):
                return
            self._start_episode()
            return

        if self.episode_active:
            moved = math.hypot(self.pose_x - self.progress_anchor_x, self.pose_y - self.progress_anchor_y)
            if moved >= self.progress_distance:
                self.progress_anchor_x = self.pose_x
                self.progress_anchor_y = self.pose_y
                self.last_progress_time_sec = self._now_sec()

    def on_cmd(self, msg: Twist) -> None:
        self.latest_linear_cmd = float(msg.linear.x)
        self.last_cmd_time_sec = self._now_sec()

    def on_safety(self, msg: Bool) -> None:
        if msg.data:
            self.last_safety_hit_sec = self._now_sec()

    def on_timer(self) -> None:
        if not self.odom_received:
            return

        now_sec = self._now_sec()
        if self.pending_reset:
            if now_sec < self.reset_ready_at_sec:
                return

            if not self._randomization_applied_for_pending_reset:
                self._apply_episode_randomization_if_enabled()
                self._randomization_applied_for_pending_reset = True

            if self.wait_for_controllers_active and not self._controllers_are_active(now_sec):
                return
            if not self._post_reset_ready_for_episode(now_sec):
                return

            self.pending_reset = False
            self._start_episode()
            self._next_episode_after_reset = False
            return

        if not self.episode_active:
            return

        if self.odom_msg_count <= self.episode_start_odom_count:
            return

        if now_sec < self.goal_guard_until_sec:
            return

        dist = math.hypot(self.goal_x - self.pose_x, self.goal_y - self.pose_y)
        if self.goal_rearm_pending:
            rearm_threshold = self.goal_tolerance + self.post_reset_goal_rearm_distance
            if dist >= rearm_threshold:
                self.goal_rearm_pending = False
                self.last_progress_time_sec = now_sec
                self.best_goal_distance = dist
                self.get_logger().info(
                    f"Episode {self.episode_id} re-armed: goal_distance={dist:.3f} m "
                    f"(threshold={rearm_threshold:.3f} m)"
                )
            elif (
                self.rearm_timeout_sec > 0.0
                and (now_sec - self.goal_rearm_started_sec) >= self.rearm_timeout_sec
                and self.fallback_all_reset_on_rearm_timeout
                and self.auto_reset_enabled
                and self.reset_mode == "model_only"
            ):
                self.get_logger().warn(
                    "Goal re-arm timeout in model_only reset mode. "
                    "Trying fallback full reset (all=true)."
                )
                self.episode_active = False
                self.goal_rearm_pending = False
                if self._reset_world(override_mode="all"):
                    self._respawn_controllers_after_all_reset_if_needed(mode="all")
                    self.pending_reset = True
                    self._randomization_applied_for_pending_reset = False
                    self._next_episode_after_reset = True
                    self.reset_ready_at_sec = now_sec + self.reset_pause_sec
                    self.reset_wait_started_sec = now_sec
                    self.pending_reset_fallback_used = True
                else:
                    self.get_logger().warn(
                        "Fallback full reset failed. Episode manager is idle until next odom cycle."
                    )
                return
            else:
                return

        if self.goal_progress_distance > 0.0 and (
            not math.isfinite(self.best_goal_distance)
            or dist <= (self.best_goal_distance - self.goal_progress_distance)
        ):
            self.best_goal_distance = dist
            self.last_progress_time_sec = now_sec

        if self.reset_on_goal and dist <= self.goal_tolerance:
            self._finish_episode("goal", dist)
            return

        if self.max_episode_sec > 0.0:
            episode_elapsed_sec = now_sec - self.episode_start_time_sec
            if episode_elapsed_sec >= self.max_episode_sec:
                self._finish_episode("timeout", dist)
                return

        if self.reset_on_stuck and self._is_stuck(now_sec):
            self._finish_episode("stuck", dist)

    def _start_episode(self) -> None:
        now_sec = self._now_sec()
        self.episode_id += 1
        self.episode_active = True
        self.episode_start_time_sec = now_sec
        self.episode_start_odom_count = self.odom_msg_count
        self.goal_guard_until_sec = now_sec + self.post_reset_goal_guard_sec
        self.last_progress_time_sec = now_sec
        self.progress_anchor_x = self.pose_x
        self.progress_anchor_y = self.pose_y
        self.best_goal_distance = math.hypot(self.goal_x - self.pose_x, self.goal_y - self.pose_y)
        self.goal_rearm_pending = False
        self.goal_rearm_started_sec = now_sec

        self.get_logger().info(
            f"Episode {self.episode_id} started: goal=({self.goal_x:.2f}, {self.goal_y:.2f})"
        )
        self._publish_episode_event(
            "episode_start",
            episode_id=self.episode_id,
            goal_x=self.goal_x,
            goal_y=self.goal_y,
            goal_tolerance=self.goal_tolerance,
        )

    def _finish_episode(self, reason: str, dist: float) -> None:
        self.episode_active = False
        self.get_logger().info(
            f"Episode {self.episode_id} finished: reason={reason}, dist_to_goal={dist:.3f} m"
        )
        self._publish_episode_event(
            "episode_end",
            episode_id=self.episode_id,
            reason=reason,
            dist_to_goal=dist,
            goal_x=self.goal_x,
            goal_y=self.goal_y,
        )
        if not self.auto_reset_enabled:
            return
        if self._reset_world():
            self._respawn_controllers_after_all_reset_if_needed(mode=self.reset_mode)
            self.pending_reset = True
            self._randomization_applied_for_pending_reset = False
            self._next_episode_after_reset = True
            now_sec = self._now_sec()
            self.reset_ready_at_sec = now_sec + self.reset_pause_sec
            self.reset_wait_started_sec = now_sec
            self.pending_reset_fallback_used = self.reset_mode == "all"
        else:
            self.get_logger().warn("World reset failed. Episode manager is idle until next odom cycle.")

    def _is_stuck(self, now_sec: float) -> bool:
        if (now_sec - self.episode_start_time_sec) < self.min_episode_sec:
            return False
        no_progress_too_long = (now_sec - self.last_progress_time_sec) >= self.stuck_timeout_sec
        if not no_progress_too_long:
            return False

        cmd_recent = (now_sec - self.last_cmd_time_sec) <= self.cmd_stale_sec
        cmd_trying_to_move = abs(self.latest_linear_cmd) >= self.stuck_min_cmd_linear
        safety_recent = (now_sec - self.last_safety_hit_sec) <= self.cmd_stale_sec

        # Consider stuck if either the planner keeps trying to move
        # or safety keeps intervening while no meaningful progress happens.
        return (cmd_recent and cmd_trying_to_move) or safety_recent

    def _reset_world(self, override_mode: str | None = None) -> bool:
        mode = self.reset_mode
        if override_mode in ("all", "model_only", "time_only"):
            mode = override_mode

        if mode == "all":
            reset_req = "reset: {all: true}"
        elif mode == "time_only":
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
            self._publish_episode_event(
                "reset_failed",
                episode_id=self.episode_id,
                reset_mode=mode,
                error=stderr if stderr else f"exit_code={result.returncode}",
            )
            return False

        self.get_logger().info("World reset requested")
        self._publish_episode_event(
            "reset_requested",
            episode_id=self.episode_id,
            reset_mode=mode,
        )
        return True

    def _respawn_controllers_after_all_reset_if_needed(self, mode: str) -> None:
        if not self.respawn_controllers_after_all_reset:
            return
        if mode != "all":
            return
        if not self.controller_params_file:
            self.get_logger().warn(
                "controller_params_file is empty. Skip controller respawn after full reset."
            )
            return

        self.get_logger().info("Respawning controllers after full reset...")
        if not self._wait_controller_manager_ready():
            self.get_logger().warn("controller_manager not ready after full reset; skip respawn.")
            return

        controllers = [
            "joint_state_broadcaster",
            "ackermann_steering_controller",
            "front_wheel_velocity_controller",
        ]
        for controller_name in controllers:
            if self._respawn_single_controller(controller_name):
                self.get_logger().info(f"Respawned controller '{controller_name}'")
            else:
                self.get_logger().warn(f"Respawn '{controller_name}' failed.")

    def _controllers_are_active(self, now_sec: float) -> bool:
        if not self.required_controllers:
            return True
        if (now_sec - self._last_controller_check_sec) < self.controllers_check_period_sec:
            return self._last_controller_check_ok
        self._last_controller_check_sec = now_sec

        states, error = self._list_controllers()
        self._last_controller_states = states
        if error:
            self._last_controller_check_ok = False
            self._log_controllers_wait(
                now_sec,
                f"list_controllers returned error: {error}",
            )
            return False

        for controller_name in self.required_controllers:
            state = states.get(controller_name, "")
            if state != "active":
                self._last_controller_check_ok = False
                self._log_controllers_wait(
                    now_sec, f"waiting controller '{controller_name}' to become active (state='{state or 'missing'}')"
                )
                return False

        self._last_controller_check_ok = True
        return True

    def _log_controllers_wait(self, now_sec: float, message: str) -> None:
        if (now_sec - self._last_controller_wait_log_sec) >= 1.0:
            self.get_logger().warn(message)
            self._last_controller_wait_log_sec = now_sec

    def _post_reset_ready_for_episode(self, now_sec: float) -> bool:
        if not self._next_episode_after_reset or self.post_reset_goal_rearm_distance <= 0.0:
            return True
        rearm_threshold = self.goal_tolerance + self.post_reset_goal_rearm_distance
        dist = math.hypot(self.goal_x - self.pose_x, self.goal_y - self.pose_y)
        if dist >= rearm_threshold:
            return True
        if self.rearm_timeout_sec <= 0.0:
            self._log_controllers_wait(
                now_sec,
                f"Waiting post-reset rearm: goal_distance={dist:.3f} m, threshold={rearm_threshold:.3f} m",
            )
            return False

        elapsed = now_sec - self.reset_wait_started_sec
        if elapsed < self.rearm_timeout_sec:
            self._log_controllers_wait(
                now_sec,
                f"Waiting post-reset rearm: goal_distance={dist:.3f} m, threshold={rearm_threshold:.3f} m",
            )
            return False

        if (
            self.fallback_all_reset_on_rearm_timeout
            and self.auto_reset_enabled
            and self.reset_mode == "model_only"
            and not self.pending_reset_fallback_used
        ):
            self.get_logger().warn(
                "Goal re-arm timeout in model_only reset mode before episode start. "
                "Trying fallback full reset (all=true)."
            )
            return self._retry_full_reset_for_rearm(now_sec)

        if (
            self.fallback_all_reset_on_rearm_timeout
            and self.auto_reset_enabled
            and self.pending_reset_fallback_used
        ):
            self.get_logger().warn(
                "Post-reset rearm still blocked after fallback reset. Retrying full reset."
            )
            return self._retry_full_reset_for_rearm(now_sec)

        return False

    def _retry_full_reset_for_rearm(self, now_sec: float) -> bool:
        if not self._reset_world(override_mode="all"):
            self.get_logger().warn("Fallback full reset failed while waiting rearm.")
            self.reset_wait_started_sec = now_sec
            return False
        self._respawn_controllers_after_all_reset_if_needed(mode="all")
        self.pending_reset = True
        self._randomization_applied_for_pending_reset = False
        self._next_episode_after_reset = True
        self.reset_ready_at_sec = now_sec + self.reset_pause_sec
        self.reset_wait_started_sec = now_sec
        self.pending_reset_fallback_used = True
        return False

    def _apply_episode_randomization_if_enabled(self) -> None:
        if not self.randomization_enabled:
            return

        start_pose = None
        if self.randomize_start_pose:
            start_pose = (
                self.rng.uniform(self.start_x_min, self.start_x_max),
                self.rng.uniform(self.start_y_min, self.start_y_max),
                self.rng.uniform(self.start_yaw_min, self.start_yaw_max),
            )
            self._set_model_pose(self.car_model_name, start_pose[0], start_pose[1], 0.06, start_pose[2])

        if self.randomize_goal:
            goal_x = self.rng.uniform(self.goal_x_min, self.goal_x_max)
            goal_y = self.rng.uniform(self.goal_y_min, self.goal_y_max)
            self.goal_x = goal_x
            self.goal_y = goal_y
            self._push_goal_to_runtime_nodes(goal_x, goal_y)

        if self.randomize_obstacles and self.obstacle_model_names:
            blocked_points: list[tuple[float, float]] = []
            if start_pose is not None:
                blocked_points.append((start_pose[0], start_pose[1]))
            blocked_points.append((self.goal_x, self.goal_y))
            for model_name in self.obstacle_model_names:
                pose = self._sample_obstacle_pose(blocked_points)
                if pose is None:
                    self.get_logger().warn(f"Failed to sample obstacle pose for {model_name}")
                    continue
                blocked_points.append((pose[0], pose[1]))
                self._set_model_pose(model_name, pose[0], pose[1], 0.25, 0.0)

        self._publish_episode_event(
            "episode_randomized",
            episode_id=self.episode_id + 1,
            goal_x=self.goal_x,
            goal_y=self.goal_y,
        )

    def _sample_obstacle_pose(
        self, blocked_points: list[tuple[float, float]]
    ) -> tuple[float, float] | None:
        for _ in range(30):
            x = self.rng.uniform(self.obstacle_x_min, self.obstacle_x_max)
            y = self.rng.uniform(self.obstacle_y_min, self.obstacle_y_max)
            if all(
                math.hypot(x - bx, y - by) >= self.randomization_min_clearance
                for (bx, by) in blocked_points
            ):
                return (x, y)
        return None

    def _set_model_pose(self, model_name: str, x: float, y: float, z: float, yaw: float) -> None:
        qz = math.sin(yaw * 0.5)
        qw = math.cos(yaw * 0.5)
        req = (
            f'name: "{model_name}", '
            f'position: {{x: {x:.4f}, y: {y:.4f}, z: {z:.4f}}}, '
            f'orientation: {{x: 0.0, y: 0.0, z: {qz:.6f}, w: {qw:.6f}}}'
        )
        cmd = [
            "gz",
            "service",
            "-s",
            self.world_set_pose_service,
            "--reqtype",
            "gz.msgs.Pose",
            "--reptype",
            "gz.msgs.Boolean",
            "--timeout",
            "2000",
            "--req",
            req,
        ]
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3.0,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            self.get_logger().warn(f"Failed set_pose for {model_name}: {exc}")
            return

        if result.returncode != 0:
            stderr = result.stderr.strip()
            self.get_logger().warn(
                f"set_pose failed for {model_name}: "
                + (stderr if stderr else f"exit_code={result.returncode}")
            )

    def _push_goal_to_runtime_nodes(self, goal_x: float, goal_y: float) -> None:
        goal_targets = [
            "hybrid_episode_manager",
            "hybrid_rule_planner",
            "hybrid_rl_planner",
            "hybrid_metrics_logger",
            "hybrid_rl_dataset_collector",
        ]
        for node_name in goal_targets:
            self._set_node_param(node_name, "goal_x", f"{goal_x:.4f}")
            self._set_node_param(node_name, "goal_y", f"{goal_y:.4f}")

    def _set_node_param(self, node_name: str, param_name: str, value: str) -> None:
        cmd = ["ros2", "param", "set", node_name, param_name, value]
        try:
            subprocess.run(cmd, capture_output=True, text=True, timeout=2.0, check=False)
        except (OSError, subprocess.SubprocessError):
            pass

    def _publish_episode_event(self, event_type: str, **payload: object) -> None:
        msg = String()
        event = {"event": event_type, "stamp_sec": self._now_sec(), **payload}
        msg.data = json.dumps(event, separators=(",", ":"))
        self.event_pub.publish(msg)

    def _wait_controller_manager_ready(self) -> bool:
        deadline = self._now_sec() + min(self.controller_manager_timeout_sec, 20.0)
        while self._now_sec() < deadline:
            _, error = self._list_controllers()
            if not error:
                return True
            time.sleep(0.3)
        return False

    def _respawn_single_controller(self, controller_name: str) -> bool:
        retries = self.controller_respawn_retries
        for attempt in range(1, retries + 1):
            states, error = self._list_controllers()
            if not error and states.get(controller_name, "") == "active":
                return True

            cmd = [
                "ros2",
                "run",
                "controller_manager",
                "spawner",
                controller_name,
                "--controller-manager",
                self.controller_manager_name,
                "--param-file",
                self.controller_params_file,
                "--controller-manager-timeout",
                f"{self.controller_manager_timeout_sec:.1f}",
                "--service-call-timeout",
                f"{self.controller_service_call_timeout_sec:.1f}",
            ]
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=self.controller_manager_timeout_sec + 10.0,
                    check=False,
                )
            except (OSError, subprocess.SubprocessError) as exc:
                self.get_logger().warn(
                    f"Respawn '{controller_name}' attempt {attempt}/{retries} failed: {exc}"
                )
                time.sleep(self.controller_respawn_retry_delay_sec)
                continue

            if result.returncode != 0:
                stderr = result.stderr.strip()
                self.get_logger().warn(
                    f"Respawn '{controller_name}' attempt {attempt}/{retries} returned "
                    + (stderr if stderr else f"exit_code={result.returncode}")
                )

            states, error = self._list_controllers()
            if not error and states.get(controller_name, "") == "active":
                return True
            time.sleep(self.controller_respawn_retry_delay_sec)
        return False

    def _list_controllers(self) -> tuple[dict[str, str], str]:
        cmd = [
            "ros2",
            "control",
            "list_controllers",
            "--controller-manager",
            self.controller_manager_name,
        ]
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=5.0,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            return {}, str(exc)

        if result.returncode != 0:
            stderr = result.stderr.strip()
            return {}, stderr if stderr else f"exit_code={result.returncode}"

        return self._parse_controller_states(result.stdout), ""

    @staticmethod
    def _parse_controller_states(text: str) -> dict[str, str]:
        states: dict[str, str] = {}
        known_states = {"active", "inactive", "unconfigured", "finalized"}
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("controller") or line.startswith("name:"):
                continue
            # Normalize common formatting variants:
            # 1) name[type] state
            # 2) name [type] state
            # 3) name type state
            normalized = line.replace("[", " [").replace("]", "] ")
            tokens = normalized.split()
            if not tokens:
                continue
            controller_name = tokens[0]
            state = ""
            for token in reversed(tokens):
                t = token.strip().strip(",").lower()
                if t in known_states:
                    state = t
                    break
            if state:
                states[controller_name] = state
        return states

    def _now_sec(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    @staticmethod
    def _ordered_range(a: float, b: float) -> tuple[float, float]:
        return (a, b) if a <= b else (b, a)


def main() -> None:
    rclpy.init()
    node = EpisodeManagerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
