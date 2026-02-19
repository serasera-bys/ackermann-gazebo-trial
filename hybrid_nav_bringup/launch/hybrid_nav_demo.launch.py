import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


def generate_launch_description():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    default_experiments_dir = os.environ.get(
        "HYBRID_NAV_EXPERIMENTS_DIR",
        os.path.join(repo_root, "experiments"),
    )
    default_metrics_output_file = os.path.join(default_experiments_dir, "latest_metrics.json")
    default_metrics_episode_summary_file = os.path.join(
        default_experiments_dir,
        "episode_metrics.jsonl",
    )
    default_dataset_output_file = os.path.join(default_experiments_dir, "rl_dataset.jsonl")
    default_rl_policy_file = os.environ.get(
        "HYBRID_NAV_POLICY_FILE",
        os.path.join(default_experiments_dir, "rl_policy.json"),
    )
    controller_params_file = os.path.join(
        get_package_share_directory("ackermann_bringup"),
        "config",
        "ros2_controllers.yaml",
    )
    goal_x = LaunchConfiguration("goal_x")
    goal_y = LaunchConfiguration("goal_y")
    reach_tolerance = LaunchConfiguration("reach_tolerance")
    enable_metrics = LaunchConfiguration("enable_metrics")
    metrics_output_file = LaunchConfiguration("metrics_output_file")
    metrics_episode_summary_file = LaunchConfiguration("metrics_episode_summary_file")
    metrics_append_episode_summary = LaunchConfiguration("metrics_append_episode_summary")
    benchmark_scenario_label = LaunchConfiguration("benchmark_scenario_label")
    enable_dataset_collection = LaunchConfiguration("enable_dataset_collection")
    dataset_output_file = LaunchConfiguration("dataset_output_file")
    dataset_cmd_topic = LaunchConfiguration("dataset_cmd_topic")
    dataset_flush_every = LaunchConfiguration("dataset_flush_every")
    auto_reset_enabled = LaunchConfiguration("auto_reset_enabled")
    reset_on_goal = LaunchConfiguration("reset_on_goal")
    reset_on_stuck = LaunchConfiguration("reset_on_stuck")
    reset_mode = LaunchConfiguration("reset_mode")
    reset_pause_sec = LaunchConfiguration("reset_pause_sec")
    episode_progress_distance = LaunchConfiguration("episode_progress_distance")
    episode_goal_progress_distance = LaunchConfiguration("episode_goal_progress_distance")
    episode_stuck_timeout_sec = LaunchConfiguration("episode_stuck_timeout_sec")
    episode_min_episode_sec = LaunchConfiguration("episode_min_episode_sec")
    episode_max_duration_sec = LaunchConfiguration("episode_max_duration_sec")
    episode_stuck_min_cmd_linear = LaunchConfiguration("episode_stuck_min_cmd_linear")
    episode_cmd_stale_sec = LaunchConfiguration("episode_cmd_stale_sec")
    episode_event_topic = LaunchConfiguration("episode_event_topic")
    episode_wait_for_controllers_active = LaunchConfiguration("episode_wait_for_controllers_active")
    episode_controllers_check_period_sec = LaunchConfiguration("episode_controllers_check_period_sec")
    episode_post_reset_goal_guard_sec = LaunchConfiguration("episode_post_reset_goal_guard_sec")
    episode_post_reset_goal_rearm_distance = LaunchConfiguration(
        "episode_post_reset_goal_rearm_distance"
    )
    episode_rearm_timeout_sec = LaunchConfiguration("episode_rearm_timeout_sec")
    episode_fallback_all_reset_on_rearm_timeout = LaunchConfiguration(
        "episode_fallback_all_reset_on_rearm_timeout"
    )
    episode_randomization_enabled = LaunchConfiguration("episode_randomization_enabled")
    episode_random_seed = LaunchConfiguration("episode_random_seed")
    episode_randomize_start_pose = LaunchConfiguration("episode_randomize_start_pose")
    episode_randomize_goal = LaunchConfiguration("episode_randomize_goal")
    episode_randomize_obstacles = LaunchConfiguration("episode_randomize_obstacles")
    stop_distance = LaunchConfiguration("stop_distance")
    hard_stop_distance = LaunchConfiguration("hard_stop_distance")
    avoid_distance = LaunchConfiguration("avoid_distance")
    avoid_gain = LaunchConfiguration("avoid_gain")
    reverse_enabled = LaunchConfiguration("reverse_enabled")
    reverse_trigger_distance = LaunchConfiguration("reverse_trigger_distance")
    reverse_speed = LaunchConfiguration("reverse_speed")
    reverse_turn_speed = LaunchConfiguration("reverse_turn_speed")
    reverse_duration_sec = LaunchConfiguration("reverse_duration_sec")
    escape_forward_duration_sec = LaunchConfiguration("escape_forward_duration_sec")
    escape_forward_speed = LaunchConfiguration("escape_forward_speed")
    reverse_turn_mode = LaunchConfiguration("reverse_turn_mode")
    rl_policy_file = LaunchConfiguration("rl_policy_file")
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
            "reach_tolerance": reach_tolerance,
            "output_topic": "/cmd_vel_raw",
            "odom_topic": odom_topic,
            "scan_topic": "/scan",
            "avoid_distance": avoid_distance,
            "avoid_gain": avoid_gain,
            "reverse_enabled": reverse_enabled,
            "reverse_trigger_distance": reverse_trigger_distance,
            "reverse_speed": reverse_speed,
            "reverse_turn_speed": reverse_turn_speed,
            "reverse_duration_sec": reverse_duration_sec,
            "escape_forward_duration_sec": escape_forward_duration_sec,
            "escape_forward_speed": escape_forward_speed,
            "reverse_turn_mode": reverse_turn_mode,
            "max_linear_speed": 0.8,
            "max_angular_speed": 0.7,
            "use_episode_events": True,
            "episode_event_topic": episode_event_topic,
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
            "reach_tolerance": reach_tolerance,
            "output_topic": "/cmd_vel_raw",
            "odom_topic": odom_topic,
            "scan_topic": "/scan",
            "policy_file": rl_policy_file,
            "avoid_distance": avoid_distance,
            "max_linear_speed": 0.8,
            "max_angular_speed": 0.7,
            "use_episode_events": True,
            "episode_event_topic": episode_event_topic,
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
            "stop_distance": stop_distance,
            "hard_stop_distance": hard_stop_distance,
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
            "episode_event_topic": episode_event_topic,
            "use_episode_events": True,
            "output_file": metrics_output_file,
            "episode_summary_file": metrics_episode_summary_file,
            "append_episode_summary": metrics_append_episode_summary,
            "planner_mode_label": planner_mode,
            "scenario_label": benchmark_scenario_label,
            "goal_x": goal_x,
            "goal_y": goal_y,
            "goal_tolerance": reach_tolerance,
            "collision_distance": 0.2,
        }],
    )

    dataset_collector = Node(
        package="hybrid_nav_rl_planner",
        executable="dataset_collector_node",
        name="hybrid_rl_dataset_collector",
        output="screen",
        condition=IfCondition(enable_dataset_collection),
        parameters=[{
            "output_file": dataset_output_file,
            "goal_x": goal_x,
            "goal_y": goal_y,
            "goal_tolerance": reach_tolerance,
            "odom_topic": odom_topic,
            "scan_topic": "/scan",
            "cmd_topic": dataset_cmd_topic,
            "flush_every": dataset_flush_every,
            "scenario_label": benchmark_scenario_label,
            "use_episode_events": True,
            "episode_event_topic": episode_event_topic,
            # Reset handled by episode_manager when using main launch.
            "auto_reset": False,
        }],
    )

    episode_manager = Node(
        package="hybrid_nav_bringup",
        executable="episode_manager_node",
        name="hybrid_episode_manager",
        output="screen",
        parameters=[{
            "auto_reset_enabled": auto_reset_enabled,
            "reset_on_goal": reset_on_goal,
            "reset_on_stuck": reset_on_stuck,
            "reset_mode": reset_mode,
            "reset_pause_sec": reset_pause_sec,
            "goal_x": goal_x,
            "goal_y": goal_y,
            "goal_tolerance": reach_tolerance,
            "odom_topic": odom_topic,
            "cmd_topic": "/cmd_vel",
            "safety_topic": "/safety_layer/intervention",
            "episode_event_topic": episode_event_topic,
            "world_control_service": "/world/default/control",
            "world_set_pose_service": "/world/default/set_pose",
            "progress_distance": episode_progress_distance,
            "goal_progress_distance": episode_goal_progress_distance,
            "stuck_timeout_sec": episode_stuck_timeout_sec,
            "min_episode_sec": episode_min_episode_sec,
            "max_episode_sec": episode_max_duration_sec,
            "stuck_min_cmd_linear": episode_stuck_min_cmd_linear,
            "cmd_stale_sec": episode_cmd_stale_sec,
            "wait_for_controllers_active": episode_wait_for_controllers_active,
            "controllers_check_period_sec": episode_controllers_check_period_sec,
            "post_reset_goal_guard_sec": episode_post_reset_goal_guard_sec,
            "post_reset_goal_rearm_distance": episode_post_reset_goal_rearm_distance,
            "rearm_timeout_sec": episode_rearm_timeout_sec,
            "fallback_all_reset_on_rearm_timeout": episode_fallback_all_reset_on_rearm_timeout,
            "respawn_controllers_after_all_reset": True,
            "controller_manager_name": "/controller_manager",
            "controller_params_file": controller_params_file,
            "controller_manager_timeout_sec": 120.0,
            "controller_service_call_timeout_sec": 30.0,
            "randomization_enabled": episode_randomization_enabled,
            "random_seed": episode_random_seed,
            "randomize_start_pose": episode_randomize_start_pose,
            "randomize_goal": episode_randomize_goal,
            "randomize_obstacles": episode_randomize_obstacles,
        }],
    )

    return LaunchDescription([
        DeclareLaunchArgument("goal_x", default_value="8.0"),
        DeclareLaunchArgument("goal_y", default_value="0.0"),
        DeclareLaunchArgument("reach_tolerance", default_value="0.25"),
        DeclareLaunchArgument(
            "metrics_output_file",
            default_value=default_metrics_output_file,
        ),
        DeclareLaunchArgument(
            "metrics_episode_summary_file",
            default_value=default_metrics_episode_summary_file,
        ),
        DeclareLaunchArgument("metrics_append_episode_summary", default_value="false"),
        DeclareLaunchArgument("benchmark_scenario_label", default_value="default"),
        DeclareLaunchArgument("enable_dataset_collection", default_value="false"),
        DeclareLaunchArgument(
            "dataset_output_file",
            default_value=default_dataset_output_file,
        ),
        DeclareLaunchArgument("dataset_cmd_topic", default_value="/cmd_vel"),
        DeclareLaunchArgument("dataset_flush_every", default_value="50"),
        DeclareLaunchArgument("auto_reset_enabled", default_value="true"),
        DeclareLaunchArgument("reset_on_goal", default_value="true"),
        DeclareLaunchArgument("reset_on_stuck", default_value="true"),
        DeclareLaunchArgument("reset_mode", default_value="model_only"),
        DeclareLaunchArgument("reset_pause_sec", default_value="1.0"),
        DeclareLaunchArgument("episode_progress_distance", default_value="0.30"),
        DeclareLaunchArgument("episode_goal_progress_distance", default_value="0.12"),
        DeclareLaunchArgument("episode_stuck_timeout_sec", default_value="5.0"),
        DeclareLaunchArgument("episode_min_episode_sec", default_value="2.0"),
        DeclareLaunchArgument("episode_max_duration_sec", default_value="0.0"),
        DeclareLaunchArgument("episode_stuck_min_cmd_linear", default_value="0.10"),
        DeclareLaunchArgument("episode_cmd_stale_sec", default_value="1.0"),
        DeclareLaunchArgument("episode_event_topic", default_value="/hybrid_nav/episode_event"),
        DeclareLaunchArgument("episode_wait_for_controllers_active", default_value="true"),
        DeclareLaunchArgument("episode_controllers_check_period_sec", default_value="0.5"),
        DeclareLaunchArgument("episode_post_reset_goal_guard_sec", default_value="1.0"),
        DeclareLaunchArgument("episode_post_reset_goal_rearm_distance", default_value="0.8"),
        DeclareLaunchArgument("episode_rearm_timeout_sec", default_value="2.5"),
        DeclareLaunchArgument("episode_fallback_all_reset_on_rearm_timeout", default_value="true"),
        DeclareLaunchArgument("episode_randomization_enabled", default_value="false"),
        DeclareLaunchArgument("episode_random_seed", default_value="-1"),
        DeclareLaunchArgument("episode_randomize_start_pose", default_value="true"),
        DeclareLaunchArgument("episode_randomize_goal", default_value="true"),
        DeclareLaunchArgument("episode_randomize_obstacles", default_value="true"),
        DeclareLaunchArgument("stop_distance", default_value="0.60"),
        DeclareLaunchArgument("hard_stop_distance", default_value="0.35"),
        DeclareLaunchArgument("avoid_distance", default_value="1.00"),
        DeclareLaunchArgument("avoid_gain", default_value="1.20"),
        DeclareLaunchArgument("reverse_enabled", default_value="true"),
        DeclareLaunchArgument("reverse_trigger_distance", default_value="0.28"),
        DeclareLaunchArgument("reverse_speed", default_value="0.30"),
        DeclareLaunchArgument("reverse_turn_speed", default_value="0.55"),
        DeclareLaunchArgument("reverse_duration_sec", default_value="1.2"),
        DeclareLaunchArgument("escape_forward_duration_sec", default_value="1.0"),
        DeclareLaunchArgument("escape_forward_speed", default_value="0.30"),
        DeclareLaunchArgument("reverse_turn_mode", default_value="auto"),
        DeclareLaunchArgument(
            "rl_policy_file",
            default_value=default_rl_policy_file,
        ),
        DeclareLaunchArgument("enable_metrics", default_value="true"),
        DeclareLaunchArgument("odom_topic", default_value="/ackermann_steering_controller/odometry"),
        DeclareLaunchArgument("planner_mode", default_value="rule"),
        ackermann_launch,
        rule_planner,
        rl_planner,
        safety_layer,
        metrics_logger,
        dataset_collector,
        episode_manager,
    ])
