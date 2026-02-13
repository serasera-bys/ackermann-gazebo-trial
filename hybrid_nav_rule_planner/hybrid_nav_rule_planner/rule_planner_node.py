import math

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node


class RulePlannerNode(Node):
    def __init__(self) -> None:
        super().__init__("hybrid_rule_planner")

        self.declare_parameter("rate_hz", 20.0)
        self.declare_parameter("goal_x", 8.0)
        self.declare_parameter("goal_y", 0.0)
        self.declare_parameter("reach_tolerance", 0.4)
        self.declare_parameter("max_linear_speed", 0.8)
        self.declare_parameter("max_angular_speed", 0.7)
        self.declare_parameter("k_linear", 0.8)
        self.declare_parameter("k_heading", 1.5)
        self.declare_parameter("output_topic", "/cmd_vel_raw")
        self.declare_parameter("odom_topic", "/odom")

        self.rate_hz = float(self.get_parameter("rate_hz").value)
        self.goal_x = float(self.get_parameter("goal_x").value)
        self.goal_y = float(self.get_parameter("goal_y").value)
        self.reach_tolerance = float(self.get_parameter("reach_tolerance").value)
        self.max_linear_speed = float(self.get_parameter("max_linear_speed").value)
        self.max_angular_speed = float(self.get_parameter("max_angular_speed").value)
        self.k_linear = float(self.get_parameter("k_linear").value)
        self.k_heading = float(self.get_parameter("k_heading").value)
        self.output_topic = str(self.get_parameter("output_topic").value)
        self.odom_topic = str(self.get_parameter("odom_topic").value)

        self.pose_x = 0.0
        self.pose_y = 0.0
        self.yaw = 0.0
        self.odom_received = False

        self.create_subscription(Odometry, self.odom_topic, self.on_odom, 10)
        self.cmd_pub = self.create_publisher(Twist, self.output_topic, 10)
        period = 1.0 / max(self.rate_hz, 1.0)
        self.create_timer(period, self.on_tick)

        self.get_logger().info(
            f"Rule planner started: goal=({self.goal_x:.2f}, {self.goal_y:.2f}), "
            f"output={self.output_topic}"
        )

    def on_odom(self, msg: Odometry) -> None:
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        self.pose_x = p.x
        self.pose_y = p.y
        self.yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z),
        )
        self.odom_received = True

    def on_tick(self) -> None:
        cmd = Twist()
        if not self.odom_received:
            self.cmd_pub.publish(cmd)
            return

        dx = self.goal_x - self.pose_x
        dy = self.goal_y - self.pose_y
        dist = math.hypot(dx, dy)
        if dist < self.reach_tolerance:
            self.cmd_pub.publish(cmd)
            return

        heading_target = math.atan2(dy, dx)
        heading_error = self._wrap_to_pi(heading_target - self.yaw)

        cmd.linear.x = max(-self.max_linear_speed, min(self.max_linear_speed, self.k_linear * dist))
        cmd.angular.z = max(
            -self.max_angular_speed,
            min(self.max_angular_speed, self.k_heading * heading_error),
        )
        self.cmd_pub.publish(cmd)

    @staticmethod
    def _wrap_to_pi(angle: float) -> float:
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle


def main() -> None:
    rclpy.init()
    node = RulePlannerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()

