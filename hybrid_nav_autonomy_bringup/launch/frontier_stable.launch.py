import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    bringup_share = get_package_share_directory("hybrid_nav_autonomy_bringup")
    ackermann_share = get_package_share_directory("ackermann_bringup")
    nav2_share = get_package_share_directory("nav2_bringup")

    world = LaunchConfiguration("world")
    nav2_params = LaunchConfiguration("nav2_params")
    use_rviz = LaunchConfiguration("use_rviz")
    safety_allow_reverse = LaunchConfiguration("safety_allow_reverse")
    safety_max_reverse_speed = LaunchConfiguration("safety_max_reverse_speed")
    scan_input_topic = LaunchConfiguration("scan_input_topic")
    scan_nav_topic = LaunchConfiguration("scan_nav_topic")
    use_scan_retimestamp = LaunchConfiguration("use_scan_retimestamp")
    scan_stamp_offset_sec = LaunchConfiguration("scan_stamp_offset_sec")
    scan_max_input_age_sec = LaunchConfiguration("scan_max_input_age_sec")
    gz_headless = LaunchConfiguration("gz_headless")
    enable_camera_bridge = LaunchConfiguration("enable_camera_bridge")

    ackermann_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(ackermann_share, "launch", "ackermann_demo.launch.py")
        ),
        launch_arguments={
            "enable_planner": "false",
            "enable_cmd_vel_adapter": "true",
            "odom_topic": "/ackermann_steering_controller/odometry",
            "world": world,
            "cmd_vel_input_topic": "/cmd_vel_safe",
            "use_sim_time": "True",
            "gz_headless": gz_headless,
            "enable_camera_bridge": enable_camera_bridge,
        }.items(),
    )

    nav2_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(nav2_share, "launch", "bringup_launch.py")
        ),
        launch_arguments={
            "slam": "True",
            "use_localization": "True",
            "use_sim_time": "True",
            "autostart": "true",
            "use_composition": "False",
            "use_respawn": "True",
            "params_file": nav2_params,
        }.items(),
    )

    scan_retimestamp = Node(
        package="hybrid_nav_autonomy_bringup",
        executable="scan_retimestamp_node",
        name="scan_retimestamp",
        output="screen",
        parameters=[{
            "use_sim_time": True,
            "input_topic": scan_input_topic,
            "output_topic": scan_nav_topic,
            "restamp_enabled": use_scan_retimestamp,
            "stamp_offset_sec": scan_stamp_offset_sec,
            "max_input_age_sec": scan_max_input_age_sec,
        }],
    )

    safety_layer = Node(
        package="hybrid_nav_safety_layer",
        executable="safety_layer_node",
        name="semantic_safety_layer",
        output="screen",
        parameters=[{
            "use_sim_time": True,
            "input_topic": "/cmd_vel",
            "output_topic": "/cmd_vel_safe",
            "scan_topic": scan_nav_topic,
            "stop_distance": 0.50,
            "hard_stop_distance": 0.30,
            "scan_timeout_sec": 0.5,
            "max_linear_speed": 1.0,
            "max_reverse_speed": safety_max_reverse_speed,
            "max_angular_speed": 0.8,
            "allow_reverse": safety_allow_reverse,
            "front_collision_half_angle_rad": 0.45,
            "rear_collision_half_angle_rad": 0.45,
        }],
    )

    frontier_extractor = Node(
        package="hybrid_nav_frontier_explorer",
        executable="frontier_extractor_node",
        name="frontier_extractor",
        output="screen",
        parameters=[{
            "use_sim_time": True,
            "map_topic": "/map",
            "odom_topic": "/ackermann_steering_controller/odometry",
            "status_topic": "/exploration/status",
            "candidates_topic": "/exploration/frontier_candidates",
            "candidates_json_topic": "/exploration/frontier_candidates_json",
            "markers_topic": "/exploration/frontier_candidates_markers",
            "publish_rate_hz": 1.0,
            "min_cluster_size": 3,
            "max_candidates": 30,
            "require_reachable": True,
            "clearance_cells": 5,
            "goal_backoff_cells": 6,
            "min_goal_dist": 0.90,
            "max_goal_dist": 4.5,
            "failed_goal_blacklist_radius": 1.6,
            "failed_goal_blacklist_ttl_sec": 180.0,
        }],
    )

    exploration_manager = Node(
        package="hybrid_nav_frontier_explorer",
        executable="exploration_manager_node",
        name="exploration_manager",
        output="screen",
        parameters=[{
            "use_sim_time": True,
            "selected_goal_topic": "/exploration/disabled_selected_goal",
            "frontier_candidates_topic": "/exploration/frontier_candidates",
            "status_topic": "/exploration/status",
            "cmd_vel_topic": "/cmd_vel",
            "goal_timeout_sec": 50.0,
            "goal_reissue_cooldown_sec": 5.0,
            "use_frontier_fallback": True,
            "enable_adaptive_frontier_fallback": True,
            "failure_streak_forced_fallback": 3,
            "forced_fallback_duration_sec": 20.0,
            "fallback_goal_min_separation": 1.0,
            "max_fallback_goal_distance": 6.0,
            "enable_stuck_cancel": True,
            "stuck_no_progress_timeout_sec": 22.0,
            "stuck_no_progress_timeout_recovery_sec": 28.0,
            "stuck_progress_distance": 0.03,
            "stuck_goal_progress_epsilon": 0.03,
            "stuck_rotation_angular_threshold": 0.30,
            "stuck_rotation_linear_max": 0.06,
            "failed_goal_blacklist_radius": 1.6,
            "failed_goal_blacklist_ttl_sec": 180.0,
            "nav_auto_startup_if_unavailable": True,
            "nav_startup_retry_sec": 12.0,
        }],
    )

    rviz = Node(
        package="rviz2",
        executable="rviz2",
        name="semantic_rviz",
        output="screen",
        condition=IfCondition(use_rviz),
        arguments=[
            "-d",
            os.path.join(bringup_share, "rviz", "semantic_explorer.rviz"),
        ],
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            "world",
            default_value=os.path.join(bringup_share, "worlds", "house_explorer_world.sdf"),
        ),
        DeclareLaunchArgument(
            "nav2_params",
            default_value=os.path.join(bringup_share, "config", "nav2_frontier_stable.yaml"),
        ),
        DeclareLaunchArgument("safety_allow_reverse", default_value="false"),
        DeclareLaunchArgument("safety_max_reverse_speed", default_value="0.08"),
        DeclareLaunchArgument("scan_input_topic", default_value="/scan"),
        DeclareLaunchArgument("scan_nav_topic", default_value="/scan_nav"),
        DeclareLaunchArgument("use_scan_retimestamp", default_value="true"),
        DeclareLaunchArgument("scan_stamp_offset_sec", default_value="0.0"),
        DeclareLaunchArgument("scan_max_input_age_sec", default_value="0.25"),
        DeclareLaunchArgument("use_rviz", default_value="false"),
        DeclareLaunchArgument("gz_headless", default_value="true"),
        DeclareLaunchArgument("enable_camera_bridge", default_value="false"),
        ackermann_launch,
        scan_retimestamp,
        nav2_launch,
        safety_layer,
        frontier_extractor,
        exploration_manager,
        rviz,
    ])
