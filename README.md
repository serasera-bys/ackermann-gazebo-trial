# Hybrid Autonomous Mobile Robot (ROS2)

A focused ROS2 project that compares a rule-based planner vs an RL-stub planner under a safety layer, using the Ackermann simulation stack.

## What it does

- Rule-based local planner publishes /cmd_vel_raw toward a goal
- RL policy planner publishes /cmd_vel_raw with the same interface
- Safety layer filters /cmd_vel_raw into /cmd_vel
- Ackermann control adapter drives the simulated robot
- Metrics logger writes run stats to JSON

## Architecture

goal -> planner (/cmd_vel_raw) -> safety layer (/cmd_vel) -> ackermann control adapter -> ackermann controller

## Packages

- hybrid_nav_bringup: launch orchestration
- hybrid_nav_rule_planner: rule-based goal-seeking planner
- hybrid_nav_rl_planner: RL stub pipeline (collector + trainer + policy planner)
- hybrid_nav_safety_layer: command guard and obstacle stop
- hybrid_nav_metrics: runtime metrics logger

## Quick start

```bash
cd /home/bernard/ros2_ws
colcon build --packages-select \
  hybrid_nav_bringup \
  hybrid_nav_rule_planner \
  hybrid_nav_rl_planner \
  hybrid_nav_safety_layer \
  hybrid_nav_metrics \
  ackermann_bringup \
  ackermann_control

source /opt/ros/jazzy/setup.bash
source install/setup.bash
ros2 launch hybrid_nav_bringup hybrid_nav_demo.launch.py
```

Switch planner and set a goal:

```bash
ros2 launch hybrid_nav_bringup hybrid_nav_demo.launch.py planner_mode:=rl goal_x:=6.0 goal_y:=2.0
```

Main launch now includes an episode manager that can auto-reset on goal/stuck:

```bash
ros2 launch hybrid_nav_bringup hybrid_nav_demo.launch.py \
  auto_reset_enabled:=true reset_mode:=model_only
```

You can also run collection + reset in one launch:

```bash
ros2 launch hybrid_nav_bringup hybrid_nav_demo.launch.py \
  planner_mode:=rule goal_x:=6.0 goal_y:=2.0 reach_tolerance:=0.15 \
  auto_reset_enabled:=true reset_on_goal:=true reset_on_stuck:=true \
  enable_dataset_collection:=true dataset_cmd_topic:=/cmd_vel
```

Recommended stable reset settings (fast model reset + fallback full reset + controller re-spawn):

```bash
ros2 launch hybrid_nav_bringup hybrid_nav_demo.launch.py \
  planner_mode:=rule goal_x:=6.0 goal_y:=2.0 reach_tolerance:=0.15 \
  auto_reset_enabled:=true reset_on_goal:=true reset_on_stuck:=true \
  reset_mode:=model_only reset_pause_sec:=0.6 \
  episode_post_reset_goal_rearm_distance:=1.0 \
  episode_rearm_timeout_sec:=2.5 \
  episode_fallback_all_reset_on_rearm_timeout:=true \
  enable_dataset_collection:=true dataset_cmd_topic:=/cmd_vel
```

Expected fallback logs:
- `Respawning controllers after full reset...`
- `Respawned controller 'joint_state_broadcaster'`
- `Respawned controller 'ackermann_steering_controller'`
- `Respawned controller 'front_wheel_velocity_controller'`

## RL Pipeline (offline reward-based)

1) Collect demonstrations from rule planner:

```bash
ros2 run hybrid_nav_rl_planner dataset_collector_node --ros-args \
  -p goal_x:=6.0 -p goal_y:=2.0 \
  -p goal_tolerance:=0.15 \
  -p auto_reset:=true \
  -p stuck_timeout_sec:=4.0
```

Keep collector running while you run rule planner launch in another terminal.
With `auto_reset:=true`, collector will end an episode on goal/stuck and call world reset automatically.
Default `reset_mode` is `model_only` so controllers stay alive between episodes.

2) Train reward-based policy from dataset:

```bash
ros2 run hybrid_nav_rl_planner train_stub \
  --dataset /home/bernard/ros2_ws/src/hybrid_nav_robot/experiments/rl_dataset.jsonl \
  --output /home/bernard/ros2_ws/src/hybrid_nav_robot/experiments/rl_policy.json \
  --reward-progress-scale 6.0 \
  --reward-goal-bonus 5.0 \
  --penalty-collision 2.0 \
  --penalty-stuck-step 0.25 \
  --penalty-stuck-terminal 3.0
```

3) Run RL mode with trained policy:

```bash
ros2 launch hybrid_nav_bringup hybrid_nav_demo.launch.py \
  planner_mode:=rl goal_x:=6.0 goal_y:=2.0 \
  rl_policy_file:=/home/bernard/ros2_ws/src/hybrid_nav_robot/experiments/rl_policy.json
```

If the rule planner gets boxed-in, enable reverse recovery tuning:

```bash
ros2 launch hybrid_nav_bringup hybrid_nav_demo.launch.py \
  planner_mode:=rule goal_x:=6.0 goal_y:=2.0 \
  reverse_enabled:=true reverse_trigger_distance:=0.28 \
  reverse_speed:=0.30 reverse_turn_speed:=0.55 reverse_duration_sec:=1.2 \
  escape_forward_duration_sec:=1.0 escape_forward_speed:=0.30 \
  reverse_turn_mode:=left
```

## How to know it reached the goal (no RViz)

- Watch metrics JSON:

```bash
tail -f /home/bernard/ros2_ws/src/hybrid_nav_robot/experiments/latest_metrics.json
```

Look for `goal_reached: true` and `goal_distance_m` near zero.

- Or echo odometry:

```bash
ros2 topic echo /ackermann_steering_controller/odometry
```

## Metrics output

Metrics are written to:

`/home/bernard/ros2_ws/src/hybrid_nav_robot/experiments/latest_metrics.json`

Fields include:
- goal_distance_m, goal_reached, time_to_goal_sec
- distance_traveled_m, max_linear_cmd_mps, max_angular_cmd_rps
- min_obstacle_range_m, collision_flag, safety_interventions

## Benchmark runner

```bash
cd /home/bernard/ros2_ws
source /opt/ros/jazzy/setup.bash
source install/setup.bash
./src/hybrid_nav_robot/experiments/run_benchmark.sh
```

The runner executes:
- scenarios: `easy`, `medium`, `cluttered`
- planners: `rule`, `rl`
- episodes: `20` per scenario/planner by default (`EPISODES_PER_RUN` override supported)

Useful overrides:
- quick smoke run: `EPISODES_PER_RUN=3 MAX_WAIT_SEC=240 ./src/hybrid_nav_robot/experiments/run_benchmark.sh`
- resume interrupted run dir: `RESUME_RUN_DIR=/home/bernard/ros2_ws/src/hybrid_nav_robot/experiments/results/run_<timestamp> ./src/hybrid_nav_robot/experiments/run_benchmark.sh`
- allow partial run to exit `0`: `STRICT_COMPLETION=false ...`

Canonical outputs:
- `experiments/results/benchmark_summary.csv`
- `experiments/results/benchmark_summary.json`
- `experiments/results/success_rate.png`
- `experiments/results/time_to_goal.png`
- `experiments/results/safety_hits.png`

Per-run outputs are archived in:
- `experiments/results/run_<timestamp>/`

See detailed protocol:
- `docs/benchmark_protocol.md`
- `docs/results.md`

## Notes

- Training is offline and reward-based: collect first, train policy file, then run `planner_mode:=rl`.
- Runtime planner is inference-only (no online learning while Gazebo is running).
- The safety layer expects /scan. If your model does not publish LaserScan, min_obstacle_range will be -1.0 and safety will never trigger.

## Repo layout

- hybrid_nav_bringup/
- hybrid_nav_rule_planner/
- hybrid_nav_rl_planner/
- hybrid_nav_safety_layer/
- hybrid_nav_metrics/
- experiments/
- docs/
