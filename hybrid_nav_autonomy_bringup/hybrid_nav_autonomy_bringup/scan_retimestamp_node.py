from __future__ import annotations

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from sensor_msgs.msg import LaserScan


class ScanRetimestampNode(Node):
    def __init__(self) -> None:
        super().__init__("scan_retimestamp")

        self.declare_parameter("input_topic", "/scan")
        self.declare_parameter("output_topic", "/scan_nav")
        self.declare_parameter("output_frame_id", "")
        self.declare_parameter("restamp_enabled", True)
        # Optional output stamp offset. Keep default at 0.0 for stable timing.
        self.declare_parameter("stamp_offset_sec", 0.0)
        self.declare_parameter("max_input_age_sec", 0.25)

        input_topic = str(self.get_parameter("input_topic").value)
        output_topic = str(self.get_parameter("output_topic").value)
        self.output_frame_id = str(self.get_parameter("output_frame_id").value).strip()
        self.restamp_enabled = bool(self.get_parameter("restamp_enabled").value)
        self.stamp_offset_ns = int(float(self.get_parameter("stamp_offset_sec").value) * 1e9)
        self.max_input_age_sec = float(self.get_parameter("max_input_age_sec").value)
        self._last_stamp_ns = 0
        self._dropped_old_scans = 0

        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.pub = self.create_publisher(LaserScan, output_topic, qos)
        self.create_subscription(LaserScan, input_topic, self.on_scan, qos)

        mode = "restamp" if self.restamp_enabled else "passthrough"
        self.get_logger().info(f"scan_retimestamp started ({mode}): {input_topic} -> {output_topic}")

    def on_scan(self, msg: LaserScan) -> None:
        now_ns = int(self.get_clock().now().nanoseconds)
        input_ns = int(msg.header.stamp.sec) * 1_000_000_000 + int(msg.header.stamp.nanosec)
        if self.max_input_age_sec > 0.0 and input_ns > 0:
            age_sec = (now_ns - input_ns) / 1e9
            if age_sec > self.max_input_age_sec:
                self._dropped_old_scans += 1
                if (self._dropped_old_scans % 50) == 1:
                    self.get_logger().warn(
                        f"Dropping stale scan (age={age_sec:.3f}s > {self.max_input_age_sec:.3f}s)"
                    )
                return
        if self.restamp_enabled:
            out_ns = now_ns + self.stamp_offset_ns
        else:
            # Passthrough mode keeps source timestamps when valid.
            out_ns = input_ns if input_ns > 0 else now_ns
        if out_ns < 0:
            out_ns = 0

        # Keep monotonic to avoid out-of-order drops in consumers.
        if out_ns <= self._last_stamp_ns:
            out_ns = self._last_stamp_ns + 1
        self._last_stamp_ns = out_ns

        msg.header.stamp.sec = out_ns // 1_000_000_000
        msg.header.stamp.nanosec = out_ns % 1_000_000_000
        if self.output_frame_id:
            msg.header.frame_id = self.output_frame_id
        self.pub.publish(msg)


def main() -> None:
    rclpy.init()
    node = ScanRetimestampNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
