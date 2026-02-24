from __future__ import annotations

import json
import math

import rclpy
from geometry_msgs.msg import Pose, PoseArray
from nav_msgs.msg import OccupancyGrid, Odometry
from rclpy.node import Node
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray

from .frontier_utils import (
    cluster_frontiers,
    compute_frontier_mask,
    compute_reachable_free_mask,
    find_nearest_free_cell,
    has_occupied_within,
    is_point_blacklisted,
    occupancy_to_grid,
    prune_expired_points,
    world_to_grid,
)


class FrontierExtractorNode(Node):
    def __init__(self) -> None:
        super().__init__("frontier_extractor")

        self.declare_parameter("map_topic", "/map")
        self.declare_parameter("candidates_topic", "/exploration/frontier_candidates")
        self.declare_parameter("candidates_json_topic", "/exploration/frontier_candidates_json")
        self.declare_parameter("markers_topic", "/exploration/frontier_candidates_markers")
        self.declare_parameter("odom_topic", "/ackermann_steering_controller/odometry")
        self.declare_parameter("status_topic", "/exploration/status")
        self.declare_parameter("publish_rate_hz", 1.0)
        self.declare_parameter("min_cluster_size", 8)
        self.declare_parameter("max_candidates", 30)
        self.declare_parameter("require_reachable", True)
        self.declare_parameter("clearance_cells", 2)
        self.declare_parameter("goal_backoff_cells", 3)
        self.declare_parameter("min_goal_dist", 0.5)
        self.declare_parameter("max_goal_dist", 7.0)
        self.declare_parameter("failed_goal_blacklist_radius", 0.7)
        self.declare_parameter("failed_goal_blacklist_ttl_sec", 45.0)

        map_topic = str(self.get_parameter("map_topic").value)
        self.candidates_topic = str(self.get_parameter("candidates_topic").value)
        self.candidates_json_topic = str(self.get_parameter("candidates_json_topic").value)
        self.markers_topic = str(self.get_parameter("markers_topic").value)
        odom_topic = str(self.get_parameter("odom_topic").value)
        status_topic = str(self.get_parameter("status_topic").value)
        rate_hz = max(0.2, float(self.get_parameter("publish_rate_hz").value))
        self.min_cluster_size = int(self.get_parameter("min_cluster_size").value)
        self.max_candidates = int(self.get_parameter("max_candidates").value)
        self.require_reachable = bool(self.get_parameter("require_reachable").value)
        self.clearance_cells = max(0, int(self.get_parameter("clearance_cells").value))
        self.goal_backoff_cells = max(0, int(self.get_parameter("goal_backoff_cells").value))
        self.min_goal_dist = max(0.0, float(self.get_parameter("min_goal_dist").value))
        self.max_goal_dist = max(self.min_goal_dist, float(self.get_parameter("max_goal_dist").value))
        self.failed_goal_blacklist_radius = max(0.0, float(self.get_parameter("failed_goal_blacklist_radius").value))
        self.failed_goal_blacklist_ttl_sec = max(0.0, float(self.get_parameter("failed_goal_blacklist_ttl_sec").value))

        self.latest_map: OccupancyGrid | None = None
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.odom_ok = False
        self.failed_goal_blacklist: list[dict[str, float]] = []

        self.candidate_pub = self.create_publisher(PoseArray, self.candidates_topic, 10)
        self.json_pub = self.create_publisher(String, self.candidates_json_topic, 10)
        self.marker_pub = self.create_publisher(MarkerArray, self.markers_topic, 10)

        self.create_subscription(OccupancyGrid, map_topic, self.on_map, 10)
        self.create_subscription(Odometry, odom_topic, self.on_odom, 10)
        self.create_subscription(String, status_topic, self.on_status, 50)
        self.create_timer(1.0 / rate_hz, self.on_timer)

        self.get_logger().info(f"frontier_extractor started: map={map_topic}")

    def on_map(self, msg: OccupancyGrid) -> None:
        self.latest_map = msg

    def on_odom(self, msg: Odometry) -> None:
        self.robot_x = float(msg.pose.pose.position.x)
        self.robot_y = float(msg.pose.pose.position.y)
        self.odom_ok = True

    def on_status(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data)
        except json.JSONDecodeError:
            return

        event = str(payload.get("event", ""))
        if event not in ("goal_aborted", "timeout", "stuck"):
            return

        x = payload.get("target_x")
        y = payload.get("target_y")
        try:
            fx = float(x)
            fy = float(y)
        except (TypeError, ValueError):
            return
        if not math.isfinite(fx) or not math.isfinite(fy):
            return

        expire = self.get_clock().now().nanoseconds / 1e9 + self.failed_goal_blacklist_ttl_sec
        self.failed_goal_blacklist.append({"x": fx, "y": fy, "expire": expire})

    def on_timer(self) -> None:
        if self.latest_map is None:
            return

        self._prune_blacklist()
        info = self.latest_map.info
        width = int(info.width)
        height = int(info.height)
        resolution = float(info.resolution)
        origin_x = float(info.origin.position.x)
        origin_y = float(info.origin.position.y)

        grid = occupancy_to_grid(self.latest_map.data, width, height)
        frontier_mask = compute_frontier_mask(grid)
        clusters = cluster_frontiers(frontier_mask, min_cluster_size=self.min_cluster_size)

        reachable_mask = None
        robot_grid = None
        if self.require_reachable and self.odom_ok:
            robot_grid = world_to_grid(
                self.robot_x,
                self.robot_y,
                resolution,
                origin_x,
                origin_y,
                width,
                height,
            )
            if robot_grid is not None:
                start = find_nearest_free_cell(grid, robot_grid[0], robot_grid[1], max_radius=8)
                if start is not None:
                    reachable_mask = compute_reachable_free_mask(grid, start[0], start[1])

        candidates = []
        for cluster in clusters:
            rep = self._cluster_representative_cell(
                cluster,
                reachable_mask,
                resolution=resolution,
                origin_x=origin_x,
                origin_y=origin_y,
            )
            if rep is None:
                continue
            gx, gy = rep
            gx, gy = self._apply_goal_backoff(
                grid=grid,
                gx=gx,
                gy=gy,
                robot_grid=robot_grid,
                reachable_mask=reachable_mask,
            )

            if self.clearance_cells > 0 and has_occupied_within(grid, gx, gy, self.clearance_cells):
                continue

            wx = origin_x + (float(gx) + 0.5) * resolution
            wy = origin_y + (float(gy) + 0.5) * resolution
            if self._is_blacklisted(wx, wy):
                continue

            if self.odom_ok:
                dist = math.hypot(wx - self.robot_x, wy - self.robot_y)
                if dist < self.min_goal_dist or dist > self.max_goal_dist:
                    continue
            else:
                dist = 0.0

            free_gain = float(len(cluster))
            if reachable_mask is not None:
                reachable_count = sum(1 for cx, cy in cluster if reachable_mask[cy, cx] == 1)
                free_gain = float(reachable_count)

            candidates.append(
                {
                    "x": wx,
                    "y": wy,
                    "free_gain": free_gain,
                    "cluster_size": int(len(cluster)),
                    "distance_to_robot": float(dist),
                }
            )

        candidates.sort(key=lambda c: c["free_gain"], reverse=True)
        candidates = candidates[: self.max_candidates]

        pose_array = PoseArray()
        pose_array.header.stamp = self.get_clock().now().to_msg()
        pose_array.header.frame_id = self.latest_map.header.frame_id
        for cand in candidates:
            p = Pose()
            p.position.x = float(cand["x"])
            p.position.y = float(cand["y"])
            p.position.z = 0.0
            p.orientation.w = 1.0
            pose_array.poses.append(p)

        markers = MarkerArray()
        for idx, cand in enumerate(candidates):
            m = Marker()
            m.header = pose_array.header
            m.ns = "frontier_candidates"
            m.id = idx
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.pose.position.x = float(cand["x"])
            m.pose.position.y = float(cand["y"])
            m.pose.position.z = 0.05
            m.pose.orientation.w = 1.0
            m.scale.x = 0.2
            m.scale.y = 0.2
            m.scale.z = 0.2
            m.color.a = 0.9
            m.color.r = 1.0
            m.color.g = 0.5
            m.color.b = 0.1
            markers.markers.append(m)

        payload = String()
        payload.data = json.dumps(
            {
                "frame_id": pose_array.header.frame_id,
                "stamp_sec": self.get_clock().now().nanoseconds / 1e9,
                "candidates": candidates,
            }
        )

        self.candidate_pub.publish(pose_array)
        self.json_pub.publish(payload)
        self.marker_pub.publish(markers)

    def _cluster_representative_cell(
        self,
        cluster: list[tuple[int, int]],
        reachable_mask,
        resolution: float,
        origin_x: float,
        origin_y: float,
    ) -> tuple[int, int] | None:
        eligible = cluster
        if reachable_mask is not None:
            eligible = [(gx, gy) for gx, gy in cluster if reachable_mask[gy, gx] == 1]
            if not eligible:
                return None

        if not self.odom_ok:
            return eligible[0]

        # Frontier clusters around the robot can form a ring. Using centroid puts
        # the goal near the robot and causes deadlock. Pick the frontier point
        # farthest from current robot position so the target stays on the frontier.
        best = eligible[0]
        best_dist2 = -1.0
        for gx, gy in eligible:
            wx = origin_x + (float(gx) + 0.5) * resolution
            wy = origin_y + (float(gy) + 0.5) * resolution
            dx = wx - self.robot_x
            dy = wy - self.robot_y
            d2 = dx * dx + dy * dy
            if d2 > best_dist2:
                best_dist2 = d2
                best = (gx, gy)
        return best

    def _prune_blacklist(self) -> None:
        if not self.failed_goal_blacklist:
            return
        now = self.get_clock().now().nanoseconds / 1e9
        self.failed_goal_blacklist = prune_expired_points(self.failed_goal_blacklist, now)

    def _apply_goal_backoff(
        self,
        grid,
        gx: int,
        gy: int,
        robot_grid: tuple[int, int] | None,
        reachable_mask,
    ) -> tuple[int, int]:
        if self.goal_backoff_cells <= 0:
            return gx, gy
        if robot_grid is None:
            return gx, gy

        rx, ry = robot_grid
        dx = float(rx - gx)
        dy = float(ry - gy)
        norm = math.hypot(dx, dy)
        if norm < 1e-6:
            return gx, gy

        h, w = grid.shape
        step_x = dx / norm
        step_y = dy / norm

        best = (gx, gy)
        for step in range(1, self.goal_backoff_cells + 1):
            cx = int(round(gx + step_x * step))
            cy = int(round(gy + step_y * step))
            if cx < 0 or cy < 0 or cx >= w or cy >= h:
                break
            val = int(grid[cy, cx])
            if not (0 <= val <= 10):
                continue
            if reachable_mask is not None and reachable_mask[cy, cx] != 1:
                continue
            best = (cx, cy)
        return best

    def _is_blacklisted(self, x: float, y: float) -> bool:
        return is_point_blacklisted(
            self.failed_goal_blacklist,
            x,
            y,
            self.failed_goal_blacklist_radius,
        )


def main() -> None:
    rclpy.init()
    node = FrontierExtractorNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
