from __future__ import annotations

import json

import rclpy
from geometry_msgs.msg import Pose, PoseArray
from nav_msgs.msg import OccupancyGrid, Odometry
from rclpy.node import Node
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray

from .frontier_utils import cluster_centroid_world, cluster_frontiers, compute_frontier_mask, occupancy_to_grid


class FrontierExtractorNode(Node):
    def __init__(self) -> None:
        super().__init__("frontier_extractor")

        self.declare_parameter("map_topic", "/map")
        self.declare_parameter("candidates_topic", "/exploration/frontier_candidates")
        self.declare_parameter("candidates_json_topic", "/exploration/frontier_candidates_json")
        self.declare_parameter("markers_topic", "/exploration/frontier_candidates_markers")
        self.declare_parameter("odom_topic", "/ackermann_steering_controller/odometry")
        self.declare_parameter("publish_rate_hz", 1.0)
        self.declare_parameter("min_cluster_size", 8)
        self.declare_parameter("max_candidates", 30)

        map_topic = str(self.get_parameter("map_topic").value)
        self.candidates_topic = str(self.get_parameter("candidates_topic").value)
        self.candidates_json_topic = str(self.get_parameter("candidates_json_topic").value)
        self.markers_topic = str(self.get_parameter("markers_topic").value)
        odom_topic = str(self.get_parameter("odom_topic").value)
        rate_hz = max(0.2, float(self.get_parameter("publish_rate_hz").value))
        self.min_cluster_size = int(self.get_parameter("min_cluster_size").value)
        self.max_candidates = int(self.get_parameter("max_candidates").value)

        self.latest_map: OccupancyGrid | None = None
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.odom_ok = False

        self.candidate_pub = self.create_publisher(PoseArray, self.candidates_topic, 10)
        self.json_pub = self.create_publisher(String, self.candidates_json_topic, 10)
        self.marker_pub = self.create_publisher(MarkerArray, self.markers_topic, 10)

        self.create_subscription(OccupancyGrid, map_topic, self.on_map, 10)
        self.create_subscription(Odometry, odom_topic, self.on_odom, 10)
        self.create_timer(1.0 / rate_hz, self.on_timer)

        self.get_logger().info(f"frontier_extractor started: map={map_topic}")

    def on_map(self, msg: OccupancyGrid) -> None:
        self.latest_map = msg

    def on_odom(self, msg: Odometry) -> None:
        self.robot_x = float(msg.pose.pose.position.x)
        self.robot_y = float(msg.pose.pose.position.y)
        self.odom_ok = True

    def on_timer(self) -> None:
        if self.latest_map is None:
            return

        info = self.latest_map.info
        grid = occupancy_to_grid(self.latest_map.data, int(info.width), int(info.height))
        frontier_mask = compute_frontier_mask(grid)
        clusters = cluster_frontiers(frontier_mask, min_cluster_size=self.min_cluster_size)

        candidates = []
        for cluster in clusters:
            wx, wy = self._cluster_representative_world(cluster, info)
            candidates.append(
                {
                    "x": wx,
                    "y": wy,
                    "free_gain": float(len(cluster)),
                    "cluster_size": int(len(cluster)),
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

    def _cluster_representative_world(self, cluster: list[tuple[int, int]], info) -> tuple[float, float]:
        resolution = float(info.resolution)
        origin_x = float(info.origin.position.x)
        origin_y = float(info.origin.position.y)

        # If odom is unavailable, keep previous centroid behavior.
        if not self.odom_ok:
            return cluster_centroid_world(
                cluster,
                resolution=resolution,
                origin_x=origin_x,
                origin_y=origin_y,
            )

        # Frontier clusters around the robot can form a ring. Using centroid puts
        # the goal near the robot and causes deadlock. Pick the frontier point
        # farthest from current robot position so the target stays on the frontier.
        best = cluster[0]
        best_dist2 = -1.0
        for gx, gy in cluster:
            wx = origin_x + (float(gx) + 0.5) * resolution
            wy = origin_y + (float(gy) + 0.5) * resolution
            dx = wx - self.robot_x
            dy = wy - self.robot_y
            d2 = dx * dx + dy * dy
            if d2 > best_dist2:
                best_dist2 = d2
                best = (gx, gy)

        bx, by = best
        return (
            origin_x + (float(bx) + 0.5) * resolution,
            origin_y + (float(by) + 0.5) * resolution,
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
