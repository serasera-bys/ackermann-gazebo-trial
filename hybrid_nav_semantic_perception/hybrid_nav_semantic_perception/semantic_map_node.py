from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import rclpy
from nav_msgs.msg import OccupancyGrid
from rclpy.node import Node
from std_msgs.msg import String
from std_srvs.srv import Trigger
from visualization_msgs.msg import Marker, MarkerArray

from .map_utils import class_counts, update_semantic_cells


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


class SemanticMapNode(Node):
    def __init__(self) -> None:
        super().__init__("semantic_map")

        self.declare_parameter("map_topic", "/map")
        self.declare_parameter("projected_objects_topic", "/semantic/projected_objects_json")
        self.declare_parameter("semantic_grid_topic", "/semantic/grid")
        self.declare_parameter("object_markers_topic", "/semantic/object_markers")
        self.declare_parameter("object_observations_topic", "/semantic/object_observations_json")
        self.declare_parameter("export_output_dir", _default_output_dir())

        map_topic = str(self.get_parameter("map_topic").value)
        projected_topic = str(self.get_parameter("projected_objects_topic").value)
        semantic_grid_topic = str(self.get_parameter("semantic_grid_topic").value)
        markers_topic = str(self.get_parameter("object_markers_topic").value)
        observations_topic = str(self.get_parameter("object_observations_topic").value)
        self.export_output_dir = Path(str(self.get_parameter("export_output_dir").value))

        self.map_msg: OccupancyGrid | None = None
        self.semantic_data: list[int] = []
        self.object_cells: dict[int, dict[str, float | str]] = {}

        self.semantic_pub = self.create_publisher(OccupancyGrid, semantic_grid_topic, 10)
        self.marker_pub = self.create_publisher(MarkerArray, markers_topic, 10)
        self.obs_pub = self.create_publisher(String, observations_topic, 10)

        self.create_subscription(OccupancyGrid, map_topic, self.on_map, 10)
        self.create_subscription(String, projected_topic, self.on_projected, 10)
        self.create_service(Trigger, "/semantic_map/export", self.on_export)

        self.get_logger().info(
            f"semantic_map started: map={map_topic}, projected={projected_topic}, output={self.export_output_dir}"
        )

    def on_map(self, msg: OccupancyGrid) -> None:
        self.map_msg = msg
        cell_count = int(msg.info.width * msg.info.height)
        if len(self.semantic_data) != cell_count:
            self.semantic_data = [-1] * cell_count
            self.object_cells.clear()

    def on_projected(self, msg: String) -> None:
        if self.map_msg is None:
            return

        try:
            payload = json.loads(msg.data)
        except json.JSONDecodeError:
            self.get_logger().warn("Invalid projected object JSON payload")
            return

        objects = payload.get("objects", [])
        info = self.map_msg.info
        self.semantic_data, updated_cells = update_semantic_cells(
            semantic_data=self.semantic_data,
            width=int(info.width),
            height=int(info.height),
            resolution=float(info.resolution),
            origin_x=float(info.origin.position.x),
            origin_y=float(info.origin.position.y),
            objects=objects,
        )
        self.object_cells.update(updated_cells)

        semantic_msg = OccupancyGrid()
        semantic_msg.header.stamp = self.get_clock().now().to_msg()
        semantic_msg.header.frame_id = self.map_msg.header.frame_id
        semantic_msg.info = self.map_msg.info
        semantic_msg.data = list(self.semantic_data)
        self.semantic_pub.publish(semantic_msg)

        markers = self._build_markers(self.map_msg.header.frame_id)
        self.marker_pub.publish(markers)

        observations = String()
        observations.data = json.dumps(
            {
                "stamp_sec": self.get_clock().now().nanoseconds / 1e9,
                "class_counts": class_counts(self.object_cells),
                "objects": list(self.object_cells.values())[-200:],
            }
        )
        self.obs_pub.publish(observations)

    def on_export(self, _req: Trigger.Request, resp: Trigger.Response) -> Trigger.Response:
        if self.map_msg is None:
            resp.success = False
            resp.message = "Map not available yet"
            return resp

        now = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        out_dir = self.export_output_dir / f"run_{now}"
        out_dir.mkdir(parents=True, exist_ok=True)

        map_payload = {
            "stamp": now,
            "frame_id": self.map_msg.header.frame_id,
            "width": int(self.map_msg.info.width),
            "height": int(self.map_msg.info.height),
            "resolution": float(self.map_msg.info.resolution),
            "origin": {
                "x": float(self.map_msg.info.origin.position.x),
                "y": float(self.map_msg.info.origin.position.y),
            },
            "semantic_data": self.semantic_data,
            "object_cells": list(self.object_cells.values()),
        }
        output_path = out_dir / "semantic_map.json"
        output_path.write_text(json.dumps(map_payload, indent=2), encoding="utf-8")

        resp.success = True
        resp.message = str(output_path)
        self.get_logger().info(f"Exported semantic map to {output_path}")
        return resp

    def _build_markers(self, frame_id: str) -> MarkerArray:
        markers = MarkerArray()
        now = self.get_clock().now().to_msg()

        for idx, obj in enumerate(list(self.object_cells.values())[-500:]):
            marker = Marker()
            marker.header.frame_id = frame_id
            marker.header.stamp = now
            marker.ns = f"semantic/{obj.get('class_id', 'unknown')}"
            marker.id = idx
            marker.type = Marker.CUBE
            marker.action = Marker.ADD
            marker.pose.position.x = float(obj.get("map_x", 0.0))
            marker.pose.position.y = float(obj.get("map_y", 0.0))
            marker.pose.position.z = max(0.05, float(obj.get("map_z", 0.0)))
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.2
            marker.scale.y = 0.2
            marker.scale.z = 0.2

            class_id = str(obj.get("class_id", "unknown"))
            color = self._color_for_class(class_id)
            marker.color.r = color[0]
            marker.color.g = color[1]
            marker.color.b = color[2]
            marker.color.a = 0.85
            marker.lifetime.sec = 0
            markers.markers.append(marker)

        return markers

    def _color_for_class(self, class_id: str) -> tuple[float, float, float]:
        palette = [
            (0.95, 0.2, 0.2),
            (0.2, 0.9, 0.2),
            (0.2, 0.4, 0.95),
            (0.95, 0.8, 0.2),
            (0.7, 0.2, 0.9),
            (0.2, 0.85, 0.85),
        ]
        idx = abs(hash(class_id)) % len(palette)
        return palette[idx]


def main() -> None:
    rclpy.init()
    node = SemanticMapNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
