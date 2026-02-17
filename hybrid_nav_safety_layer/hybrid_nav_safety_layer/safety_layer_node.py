import math
from typing import Optional

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool


class SafetyLayerNode(Node):
    def __init__(self) -> None:
        super().__init__("hybrid_safety_layer")

        self.declare_parameter("input_topic", "/cmd_vel_raw")
        self.declare_parameter("output_topic", "/cmd_vel")
        self.declare_parameter("scan_topic", "/scan")
        self.declare_parameter("stop_distance", 0.6)
        self.declare_parameter("hard_stop_distance", 0.35)
        self.declare_parameter("scan_timeout_sec", 0.5)
        self.declare_parameter("max_linear_speed", 1.0)
        self.declare_parameter("max_angular_speed", 0.8)

        self.input_topic = str(self.get_parameter("input_topic").value)
        self.output_topic = str(self.get_parameter("output_topic").value)
        self.scan_topic = str(self.get_parameter("scan_topic").value)
        self.stop_distance = float(self.get_parameter("stop_distance").value)
        self.hard_stop_distance = float(self.get_parameter("hard_stop_distance").value)
        self.scan_timeout_sec = float(self.get_parameter("scan_timeout_sec").value)
        self.max_linear_speed = float(self.get_parameter("max_linear_speed").value)
        self.max_angular_speed = float(self.get_parameter("max_angular_speed").value)
        if self.hard_stop_distance >= self.stop_distance:
            self.get_logger().warn(
                "hard_stop_distance >= stop_distance, forcing hard_stop_distance = stop_distance - 0.05"
            )
            self.hard_stop_distance = max(0.0, self.stop_distance - 0.05)

        self.last_scan_time: Optional[rclpy.time.Time] = None
        self.last_scan_min_range: Optional[float] = None

        self.cmd_pub = self.create_publisher(Twist, self.output_topic, 10)
        self.guard_pub = self.create_publisher(Bool, "/safety_layer/intervention", 10)
        self.create_subscription(Twist, self.input_topic, self.on_raw_cmd, 10)
        # LaserScan from simulation typically uses SensorDataQoS (best effort).
        self.create_subscription(LaserScan, self.scan_topic, self.on_scan, qos_profile_sensor_data)

        self.get_logger().info(
            f"Safety layer started: {self.input_topic} -> {self.output_topic}, "
            f"stop_distance={self.stop_distance:.2f} m, hard_stop_distance={self.hard_stop_distance:.2f} m"
        )

    def on_scan(self, msg: LaserScan) -> None:
        finite_ranges = [r for r in msg.ranges if math.isfinite(r) and r >= msg.range_min]
        self.last_scan_min_range = min(finite_ranges) if finite_ranges else None
        self.last_scan_time = self.get_clock().now()

    def on_raw_cmd(self, msg: Twist) -> None:
        out = Twist()
        out.linear.x = self._clamp(msg.linear.x, -self.max_linear_speed, self.max_linear_speed)
        out.angular.z = self._clamp(msg.angular.z, -self.max_angular_speed, self.max_angular_speed)

        intervention = False
        if self._has_recent_scan() and self.last_scan_min_range is not None:
            if out.linear.x > 0.0:
                if self.last_scan_min_range <= self.hard_stop_distance:
                    out.linear.x = 0.0
                    intervention = True
                elif self.last_scan_min_range < self.stop_distance:
                    span = max(self.stop_distance - self.hard_stop_distance, 1e-3)
                    ratio = (self.last_scan_min_range - self.hard_stop_distance) / span
                    ratio = self._clamp(ratio, 0.15, 1.0)
                    out.linear.x *= ratio
                    intervention = True

        self.cmd_pub.publish(out)
        self.guard_pub.publish(Bool(data=intervention))

    def _has_recent_scan(self) -> bool:
        if self.last_scan_time is None:
            return False
        elapsed = (self.get_clock().now() - self.last_scan_time).nanoseconds / 1e9
        return elapsed <= self.scan_timeout_sec

    @staticmethod
    def _clamp(value: float, min_v: float, max_v: float) -> float:
        return max(min_v, min(max_v, value))


def main() -> None:
    rclpy.init()
    node = SafetyLayerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
