# Semantic RL Explorer Results

## Run Command

```bash
cd ~/ros2_ws
source /opt/ros/jazzy/setup.bash
source install/setup.bash
REPO_ROOT=~/ros2_ws/src/hybrid_nav_robot
ros2 launch hybrid_nav_autonomy_bringup semantic_autonomy.launch.py \
  model_path:=$REPO_ROOT/edge_vision_qos/artifacts/yolov8n.onnx \
  policy_file:=$REPO_ROOT/experiments/semantic_rl_policy.json
```

## Metrics Table (Fill per run)

| Run ID | Coverage % | Time to 80% (s) | Detection FPS | Collision-free | Notes |
|---|---:|---:|---:|---|---|
| run_YYYYMMDD_HHMMSS |  |  |  |  |  |

## Exported Artifacts

- Semantic map export service output:
  - `experiments/results/semantic_explorer/run_<timestamp>/semantic_map.json`
- Runtime metrics (auto-written):
  - `experiments/results/semantic_explorer/run_<timestamp>/metrics.json`
- RL dataset:
  - `experiments/semantic_rl_dataset.jsonl`
- RL policy:
  - `experiments/semantic_rl_policy.json`

## Acceptance Checklist

- [ ] RViz shows `/map`, `/semantic/grid`, `/semantic/debug_image`, `/semantic/object_markers`.
- [ ] Robot autonomously sends goals and moves without manual waypoint input.
- [ ] Detection loop sustained at `>=10 FPS`.
- [ ] Coverage reaches `>=80%` reachable area in `<=10 min`.
- [ ] Safety layer prevents near-collision events in at least 4/5 runs.
