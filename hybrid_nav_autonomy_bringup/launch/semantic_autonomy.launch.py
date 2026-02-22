import os
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


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


def generate_launch_description():
    bringup_share = get_package_share_directory("hybrid_nav_autonomy_bringup")
    ackermann_share = get_package_share_directory("ackermann_bringup")
    nav2_share = get_package_share_directory("nav2_bringup")

    repo_root = _detect_repo_root()
    default_model = str(repo_root / "edge_vision_qos" / "artifacts" / "yolov8n.onnx")

    world = LaunchConfiguration("world")
    nav2_params = LaunchConfiguration("nav2_params")
    model_path = LaunchConfiguration("model_path")
    policy_file = LaunchConfiguration("policy_file")
    use_rviz = LaunchConfiguration("use_rviz")
    enable_dataset_collection = LaunchConfiguration("enable_dataset_collection")

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
            "autostart": "True",
            "use_composition": "False",
            "params_file": nav2_params,
        }.items(),
    )

    safety_layer = Node(
        package="hybrid_nav_safety_layer",
        executable="safety_layer_node",
        name="semantic_safety_layer",
        output="screen",
        parameters=[{
            "input_topic": "/cmd_vel",
            "output_topic": "/cmd_vel_safe",
            "scan_topic": "/scan",
            "stop_distance": 0.6,
            "hard_stop_distance": 0.35,
            "scan_timeout_sec": 0.5,
            "max_linear_speed": 1.0,
            "max_angular_speed": 0.8,
        }],
    )

    detector = Node(
        package="hybrid_nav_semantic_perception",
        executable="semantic_detector_node",
        name="semantic_detector",
        output="screen",
        parameters=[{
            "image_topic": "/camera/rgb/image_raw",
            "detections_topic": "/semantic/detections",
            "debug_image_topic": "/semantic/debug_image",
            "model_path": model_path,
            "confidence_threshold": 0.2,
            "target_fps": 12.0,
        }],
    )

    projection = Node(
        package="hybrid_nav_semantic_perception",
        executable="semantic_projection_node",
        name="semantic_projection",
        output="screen",
        parameters=[{
            "detections_topic": "/semantic/detections",
            "depth_topic": "/camera/depth/image_raw",
            "camera_info_topic": "/camera/rgb/camera_info",
            "target_frame": "map",
            "projected_topic": "/semantic/projected_objects_json",
            "markers_topic": "/semantic/projected_markers",
        }],
    )

    semantic_map = Node(
        package="hybrid_nav_semantic_perception",
        executable="semantic_map_node",
        name="semantic_map",
        output="screen",
        parameters=[{
            "map_topic": "/map",
            "projected_objects_topic": "/semantic/projected_objects_json",
            "semantic_grid_topic": "/semantic/grid",
            "object_markers_topic": "/semantic/object_markers",
            "object_observations_topic": "/semantic/object_observations_json",
        }],
    )

    frontier_extractor = Node(
        package="hybrid_nav_frontier_explorer",
        executable="frontier_extractor_node",
        name="frontier_extractor",
        output="screen",
        parameters=[{
            "map_topic": "/map",
            "odom_topic": "/ackermann_steering_controller/odometry",
            "candidates_topic": "/exploration/frontier_candidates",
            "candidates_json_topic": "/exploration/frontier_candidates_json",
            "markers_topic": "/exploration/frontier_candidates_markers",
            "publish_rate_hz": 1.0,
            "min_cluster_size": 3,
            "max_candidates": 60,
        }],
    )

    rl_decider = Node(
        package="hybrid_nav_semantic_rl",
        executable="semantic_rl_decider_node",
        name="semantic_rl_decider",
        output="screen",
        parameters=[{
            "frontier_candidates_json_topic": "/exploration/frontier_candidates_json",
            "selected_goal_topic": "/semantic_rl/selected_goal",
            "candidate_scores_topic": "/semantic_rl/candidate_scores_json",
            "semantic_grid_topic": "/semantic/grid",
            "semantic_observations_topic": "/semantic/object_observations_json",
            "odom_topic": "/ackermann_steering_controller/odometry",
            "scan_topic": "/scan",
            "policy_file": policy_file,
            "goal_cooldown_sec": 1.0,
            "min_goal_distance": 0.35,
            "min_robot_goal_distance": 0.35,
            "allow_close_goal_fallback": True,
            "close_goal_min_distance": 0.18,
            "rule_w4": 0.25,
        }],
    )

    exploration_manager = Node(
        package="hybrid_nav_frontier_explorer",
        executable="exploration_manager_node",
        name="exploration_manager",
        output="screen",
        parameters=[{
            "selected_goal_topic": "/semantic_rl/selected_goal",
            "frontier_candidates_topic": "/exploration/frontier_candidates",
            "status_topic": "/exploration/status",
            "goal_timeout_sec": 30.0,
            "goal_reissue_cooldown_sec": 3.0,
            "use_frontier_fallback": True,
            "enable_adaptive_frontier_fallback": True,
            "failure_streak_forced_fallback": 2,
            "forced_fallback_duration_sec": 20.0,
            "fallback_goal_min_separation": 0.5,
            "max_fallback_goal_distance": 4.0,
            "enable_stuck_cancel": True,
            "stuck_no_progress_timeout_sec": 12.0,
            "stuck_progress_distance": 0.08,
        }],
    )

    dataset_collector = Node(
        package="hybrid_nav_semantic_rl",
        executable="semantic_dataset_collector_node",
        name="semantic_dataset_collector",
        output="screen",
        condition=IfCondition(enable_dataset_collection),
        parameters=[{
            "scores_topic": "/semantic_rl/candidate_scores_json",
            "status_topic": "/exploration/status",
            "scenario_label": "house_sim",
            "flush_every": 50,
        }],
    )

    run_metrics = Node(
        package="hybrid_nav_autonomy_bringup",
        executable="semantic_run_metrics_node",
        name="semantic_run_metrics",
        output="screen",
        parameters=[{
            "map_topic": "/map",
            "detections_topic": "/semantic/detections",
            "status_topic": "/exploration/status",
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
            default_value=os.path.join(bringup_share, "config", "nav2_semantic_params.yaml"),
        ),
        DeclareLaunchArgument("model_path", default_value=default_model),
        DeclareLaunchArgument(
            "policy_file",
            default_value=str(repo_root / "experiments" / "semantic_rl_policy.json"),
        ),
        DeclareLaunchArgument("enable_dataset_collection", default_value="false"),
        DeclareLaunchArgument("use_rviz", default_value="true"),
        ackermann_launch,
        nav2_launch,
        safety_layer,
        detector,
        projection,
        semantic_map,
        frontier_extractor,
        rl_decider,
        exploration_manager,
        dataset_collector,
        run_metrics,
        rviz,
    ])
