from __future__ import annotations

import json
from typing import Any

import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String
from tf2_ros import Buffer, TransformException, TransformListener
from visualization_msgs.msg import Marker, MarkerArray
from vision_msgs.msg import Detection2DArray

from .projection_utils import clamp_pixel, depth_to_camera_point, sample_depth_window, transform_point


class SemanticProjectionNode(Node):
    def __init__(self) -> None:
        super().__init__("semantic_projection")

        self.declare_parameter("detections_topic", "/semantic/detections")
        self.declare_parameter("depth_topic", "/camera/depth/image_raw")
        self.declare_parameter("camera_info_topic", "/camera/rgb/camera_info")
        self.declare_parameter("target_frame", "map")
        self.declare_parameter("projected_topic", "/semantic/projected_objects_json")
        self.declare_parameter("markers_topic", "/semantic/projected_markers")
        self.declare_parameter("max_depth_m", 8.0)

        self.target_frame = str(self.get_parameter("target_frame").value)
        self.max_depth_m = float(self.get_parameter("max_depth_m").value)

        det_topic = str(self.get_parameter("detections_topic").value)
        depth_topic = str(self.get_parameter("depth_topic").value)
        cam_info_topic = str(self.get_parameter("camera_info_topic").value)
        projected_topic = str(self.get_parameter("projected_topic").value)
        markers_topic = str(self.get_parameter("markers_topic").value)

        self.bridge = CvBridge()
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.latest_depth: Image | None = None
        self.latest_depth_array = None
        self.latest_depth_encoding = ""
        self.latest_cam_info: CameraInfo | None = None

        self.proj_pub = self.create_publisher(String, projected_topic, 10)
        self.marker_pub = self.create_publisher(MarkerArray, markers_topic, 10)

        self.create_subscription(Image, depth_topic, self.on_depth, 10)
        self.create_subscription(CameraInfo, cam_info_topic, self.on_camera_info, 10)
        self.create_subscription(Detection2DArray, det_topic, self.on_detections, 10)

        self.get_logger().info(
            f"semantic_projection started: detections={det_topic}, depth={depth_topic}, camera_info={cam_info_topic}, target_frame={self.target_frame}"
        )

    def on_depth(self, msg: Image) -> None:
        try:
            arr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        except Exception as exc:
            self.get_logger().warn(f"Failed to decode depth image: {exc}")
            return
        self.latest_depth = msg
        self.latest_depth_array = arr
        self.latest_depth_encoding = msg.encoding

    def on_camera_info(self, msg: CameraInfo) -> None:
        self.latest_cam_info = msg

    def on_detections(self, msg: Detection2DArray) -> None:
        if self.latest_depth_array is None or self.latest_cam_info is None:
            return

        source_frame = msg.header.frame_id or (self.latest_cam_info.header.frame_id or "camera_link")
        try:
            tf_msg = self.tf_buffer.lookup_transform(
                self.target_frame,
                source_frame,
                Time.from_msg(msg.header.stamp),
                timeout=Duration(seconds=0.1),
            )
        except TransformException as exc:
            self.get_logger().debug(f"TF lookup failed {source_frame}->{self.target_frame}: {exc}")
            return

        fx = float(self.latest_cam_info.k[0])
        fy = float(self.latest_cam_info.k[4])
        cx = float(self.latest_cam_info.k[2])
        cy = float(self.latest_cam_info.k[5])

        depth = self.latest_depth_array
        h, w = depth.shape[:2]
        encoding = self.latest_depth_encoding.lower()

        projected: list[dict[str, Any]] = []
        markers = MarkerArray()

        for idx, det in enumerate(msg.detections):
            u, v = clamp_pixel(det.bbox.center.position.x, det.bbox.center.position.y, w, h)
            try:
                depth_val = float(sample_depth_window(depth, u, v, window=2))
            except ValueError:
                continue

            if "16uc1" in encoding:
                depth_val /= 1000.0

            if depth_val <= 0.05 or depth_val > self.max_depth_m:
                continue

            cam_point = depth_to_camera_point(u, v, depth_val, fx, fy, cx, cy)
            tf_t = tf_msg.transform.translation
            tf_q = tf_msg.transform.rotation
            map_point = transform_point(
                cam_point,
                (float(tf_t.x), float(tf_t.y), float(tf_t.z)),
                (float(tf_q.x), float(tf_q.y), float(tf_q.z), float(tf_q.w)),
            )

            class_id = det.id if det.id else "unknown"
            score = 0.0
            if det.results:
                class_id = det.results[0].hypothesis.class_id or class_id
                score = float(det.results[0].hypothesis.score)

            projected.append(
                {
                    "class_id": class_id,
                    "score": score,
                    "map_x": float(map_point[0]),
                    "map_y": float(map_point[1]),
                    "map_z": float(map_point[2]),
                    "source_frame": source_frame,
                }
            )

            marker = Marker()
            marker.header.frame_id = self.target_frame
            marker.header.stamp = msg.header.stamp
            marker.ns = f"projection/{class_id}"
            marker.id = idx
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = float(map_point[0])
            marker.pose.position.y = float(map_point[1])
            marker.pose.position.z = max(0.05, float(map_point[2]))
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.18
            marker.scale.y = 0.18
            marker.scale.z = 0.18
            marker.color.a = 0.9
            marker.color.r = 0.1
            marker.color.g = 0.9
            marker.color.b = 0.2
            marker.lifetime.sec = 1
            markers.markers.append(marker)

        payload = String()
        payload.data = json.dumps(
            {
                "stamp_sec": Time.from_msg(msg.header.stamp).nanoseconds / 1e9,
                "target_frame": self.target_frame,
                "objects": projected,
            }
        )
        self.proj_pub.publish(payload)
        self.marker_pub.publish(markers)


def main() -> None:
    rclpy.init()
    node = SemanticProjectionNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
