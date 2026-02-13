import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


def generate_launch_description():
    goal_x = LaunchConfiguration("goal_x")
    goal_y = LaunchConfiguration("goal_y")
    enable_metrics = LaunchConfiguration("enable_metrics")
    odom_topic = LaunchConfiguration("odom_topic")
    planner_mode = LaunchConfiguration("planner_mode")
    use_rule = PythonExpression(["'", planner_mode, "' == 'rule'"])
    use_rl = PythonExpression(["'", planner_mode, "' == 'rl'"])

    ackermann_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("ackermann_bringup"),
                "launch",
                "ackermann_demo.launch.py",
            )
        ),
        launch_arguments={
            "enable_planner": "false",
            "enable_cmd_vel_adapter": "true",
            "odom_topic": odom_topic,
        }.items(),
    )

    rule_planner = Node(
        package="hybrid_nav_rule_planner",
        executable="rule_planner_node",
        name="hybrid_rule_planner",
        output="screen",
        condition=IfCondition(use_rule),
        parameters=[{
            "goal_x": goal_x,
            "goal_y": goal_y,
            "output_topic": "/cmd_vel_raw",
            "odom_topic": odom_topic,
            "max_linear_speed": 0.8,
            "max_angular_speed": 0.7,
        }],
    )

    rl_planner = Node(
        package="hybrid_nav_rl_planner",
        executable="rl_planner_node",
        name="hybrid_rl_planner",
        output="screen",
        condition=IfCondition(use_rl),
        parameters=[{
            "goal_x": goal_x,
            "goal_y": goal_y,
            "output_topic": "/cmd_vel_raw",
            "odom_topic": odom_topic,
            "max_linear_speed": 0.8,
            "max_angular_speed": 0.7,
        }],
    )

    safety_layer = Node(
        package="hybrid_nav_safety_layer",
        executable="safety_layer_node",
        name="hybrid_safety_layer",
        output="screen",
        parameters=[{
            "input_topic": "/cmd_vel_raw",
            "output_topic": "/cmd_vel",
            "scan_topic": "/scan",
            "stop_distance": 0.6,
            "scan_timeout_sec": 0.5,
            "max_linear_speed": 1.0,
            "max_angular_speed": 0.8,
        }],
    )

    metrics_logger = Node(
        package="hybrid_nav_metrics",
        executable="metrics_logger_node",
        name="hybrid_metrics_logger",
        output="screen",
        condition=IfCondition(enable_metrics),
        parameters=[{
            "odom_topic": odom_topic,
            "cmd_topic": "/cmd_vel",
            "scan_topic": "/scan",
            "safety_topic": "/safety_layer/intervention",
            "output_file": "/home/bernard/ros2_ws/src/hybrid_nav_robot/experiments/latest_metrics.json",
            "goal_x": goal_x,
            "goal_y": goal_y,
            "goal_tolerance": 0.4,
            "collision_distance": 0.2,
        }],
    )

    return LaunchDescription([
        DeclareLaunchArgument("goal_x", default_value="8.0"),
        DeclareLaunchArgument("goal_y", default_value="0.0"),
        DeclareLaunchArgument("enable_metrics", default_value="true"),
        DeclareLaunchArgument("odom_topic", default_value="/ackermann_steering_controller/odometry"),
        DeclareLaunchArgument("planner_mode", default_value="rule"),
        ackermann_launch,
        rule_planner,
        rl_planner,
        safety_layer,
        metrics_logger,
    ])
