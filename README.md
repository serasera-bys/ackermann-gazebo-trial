# Hybrid Autonomous Mobile Robot (ROS2)

A focused ROS2 project that compares a rule-based planner vs an RL placeholder planner under a safety layer, using the Ackermann simulation stack.

## What it does

- Rule-based local planner publishes /cmd_vel_raw toward a goal
- RL placeholder planner publishes /cmd_vel_raw with the same interface
- Safety layer filters /cmd_vel_raw into /cmd_vel
- Ackermann control adapter drives the simulated robot
- Metrics logger writes run stats to JSON

## Architecture

goal -> planner (/cmd_vel_raw) -> safety layer (/cmd_vel) -> ackermann control adapter -> ackermann controller

## Packages

- hybrid_nav_bringup: launch orchestration
- hybrid_nav_rule_planner: rule-based goal-seeking planner
- hybrid_nav_rl_planner: placeholder RL planner (same interface as rule planner)
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
/home/bernard/ros2_ws/src/hybrid_nav_robot/experiments/run_benchmark.sh
```

Outputs per run:

`/home/bernard/ros2_ws/src/hybrid_nav_robot/experiments/results/run_<timestamp>_mode_<planner>_goal_<x>_<y>.json`

## Notes

- The RL planner is a placeholder policy (rule-like behavior with noise). It only exists to match the /cmd_vel_raw interface for later RL integration.
- The safety layer expects /scan. If your model does not publish LaserScan, min_obstacle_range will be -1.0 and safety will never trigger.

## Repo layout

- hybrid_nav_bringup/
- hybrid_nav_rule_planner/
- hybrid_nav_rl_planner/
- hybrid_nav_safety_layer/
- hybrid_nav_metrics/
- experiments/
- docs/
