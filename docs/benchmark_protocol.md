# Benchmark Protocol

## Objective

Produce a reproducible comparison between `planner_mode:=rule` and `planner_mode:=rl`
on the same Ackermann ROS2/Gazebo stack.

## Fixed runtime settings

- `reset_mode:=model_only`
- `episode_fallback_all_reset_on_rearm_timeout:=true`
- `episode_wait_for_controllers_active:=true`
- `reset_pause_sec:=0.6`
- safety guard active (`stop_distance:=0.50`, `hard_stop_distance:=0.20`)
- collector disabled during benchmark (`enable_dataset_collection:=false`)

These are enforced by `experiments/run_benchmark.sh`.

## Scenario matrix

- `easy`: fixed start + fixed obstacles (`seed=11`)
- `medium`: randomized start/obstacles (`seed=23`)
- `cluttered`: randomized start/obstacles (`seed=31`) with more aggressive avoidance gain

Each scenario runs:
- planners: `rule`, `rl`
- episodes: 20 per planner (default, overridable via `EPISODES_PER_RUN`)

## Required metrics

- `success_rate`
- `mean_time_to_goal_sec`
- `collision_or_safety_intervention_rate`
- `mean_distance_traveled_m`

## Commands

```bash
cd /home/bernard/ros2_ws
source /opt/ros/jazzy/setup.bash
source install/setup.bash

# Optional overrides:
# EPISODES_PER_RUN=10 MAX_WAIT_SEC=240
# RESUME_RUN_DIR=/home/bernard/ros2_ws/src/hybrid_nav_robot/experiments/results/run_<timestamp>
# STRICT_COMPLETION=false
./src/hybrid_nav_robot/experiments/run_benchmark.sh
```

Behavior notes:
- By default, completed scenario/planner pairs are skipped (`SKIP_COMPLETED=true`).
- If `RESUME_RUN_DIR` is set, the script continues from that run directory.
- If any pair is incomplete, script exits non-zero unless `STRICT_COMPLETION=false`.

## Outputs

Per run (timestamped):
- `experiments/results/run_<timestamp>/raw/episodes_<scenario>_<planner>.jsonl`
- `experiments/results/run_<timestamp>/logs/launch_<scenario>_<planner>.log`
- `experiments/results/run_<timestamp>/benchmark_summary.csv`
- `experiments/results/run_<timestamp>/benchmark_summary.json`

Canonical latest outputs:
- `experiments/results/benchmark_summary.csv`
- `experiments/results/benchmark_summary.json`
- `experiments/results/success_rate.png`
- `experiments/results/time_to_goal.png`
- `experiments/results/safety_hits.png`

## Acceptance checks

- Goal-reached episode resets into next episode without freeze.
- Stuck episode resets into next episode without dead controller state.
- Rearm timeout triggers full reset fallback and controller respawn.
- RL mode publishes valid commands and keeps moving across episodes.
