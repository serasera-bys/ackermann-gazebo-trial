from __future__ import annotations

import os
import time
from pathlib import Path

import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose

from .detector_utils import build_detector


def _detect_repo_root() -> Path:
    env_root = os.environ.get("HYBRID_NAV_ROBOT_ROOT", "").strip()
    if env_root:
        path = Path(env_root).expanduser().resolve()
        if path.exists():
            return path

    here = Path(__file__).resolve()
    for parent in [here, *here.parents]:
        if (parent / "edge_vision_qos").exists() and (parent / "experiments").exists():
            return parent

    for parent in here.parents:
        if parent.name == "install":
            candidate = parent.parent / "src" / "hybrid_nav_robot"
            if candidate.exists():
                return candidate

    return Path.cwd()


def _default_model_path() -> str:
    default_path = _detect_repo_root() / "edge_vision_qos" / "artifacts" / "yolov8n.onnx"
    return str(default_path)


class SemanticDetectorNode(Node):
    def __init__(self) -> None:
        super().__init__("semantic_detector")

        self.declare_parameter("image_topic", "/camera/rgb/image_raw")
        self.declare_parameter("detections_topic", "/semantic/detections")
        self.declare_parameter("debug_image_topic", "/semantic/debug_image")
        self.declare_parameter("model_path", os.environ.get("SEMANTIC_DETECTOR_MODEL", _default_model_path()))
        self.declare_parameter("confidence_threshold", 0.25)
        self.declare_parameter("target_fps", 12.0)

        image_topic = str(self.get_parameter("image_topic").value)
        detections_topic = str(self.get_parameter("detections_topic").value)
        debug_image_topic = str(self.get_parameter("debug_image_topic").value)
        model_path = str(self.get_parameter("model_path").value)
        conf_threshold = float(self.get_parameter("confidence_threshold").value)
        target_fps = max(1.0, float(self.get_parameter("target_fps").value))

        self.min_period_sec = 1.0 / target_fps
        self.last_infer_time = 0.0
        self.bridge = CvBridge()
        self.detector = build_detector(model_path, conf_threshold)

        self.det_pub = self.create_publisher(Detection2DArray, detections_topic, 10)
        self.debug_pub = self.create_publisher(Image, debug_image_topic, 10)
        self.create_subscription(Image, image_topic, self.on_image, 10)

        self.get_logger().info(
            f"semantic_detector started: image={image_topic}, detections={detections_topic}, model={model_path}"
        )

    def on_image(self, msg: Image) -> None:
        now = time.monotonic()
        if (now - self.last_infer_time) < self.min_period_sec:
            return
        self.last_infer_time = now

        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as exc:
            self.get_logger().warn(f"Failed to decode image: {exc}")
            return

        detections = self.detector.infer(frame)

        arr = Detection2DArray()
        arr.header = msg.header

        for det in detections:
            item = Detection2D()
            item.header = msg.header
            item.id = det.class_id
            item.bbox.center.position.x = float((det.x1 + det.x2) * 0.5)
            item.bbox.center.position.y = float((det.y1 + det.y2) * 0.5)
            item.bbox.center.theta = 0.0
            item.bbox.size_x = float(max(1.0, det.x2 - det.x1))
            item.bbox.size_y = float(max(1.0, det.y2 - det.y1))
            hyp = ObjectHypothesisWithPose()
            hyp.hypothesis.class_id = det.class_id
            hyp.hypothesis.score = float(det.score)
            item.results.append(hyp)
            arr.detections.append(item)

        self.det_pub.publish(arr)

        annotated = frame.copy()
        for det in detections:
            p1 = (int(det.x1), int(det.y1))
            p2 = (int(det.x2), int(det.y2))
            cv2.rectangle(annotated, p1, p2, (0, 255, 0), 2)
            cv2.putText(
                annotated,
                f"{det.class_id}:{det.score:.2f}",
                (p1[0], max(0, p1[1] - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )

        debug_msg = self.bridge.cv2_to_imgmsg(annotated, encoding="bgr8")
        debug_msg.header = msg.header
        self.debug_pub.publish(debug_msg)


def main() -> None:
    rclpy.init()
    node = SemanticDetectorNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
