# Semantic RL Explorer Architecture

## Dataflow

1. Gazebo sensors publish:
   - `/scan`
   - `/camera/rgb/image_raw`
   - `/camera/depth/image_raw`
   - `/camera/rgb/camera_info`
2. `slam_toolbox` builds `/map`.
3. `semantic_detector_node` publishes:
   - `/semantic/detections`
   - `/semantic/debug_image`
4. `semantic_projection_node` projects depth + detections to map frame and publishes:
   - `/semantic/projected_objects_json`
5. `semantic_map_node` fuses observations and publishes:
   - `/semantic/grid`
   - `/semantic/object_markers`
   - `/semantic/object_observations_json`
6. `frontier_extractor_node` extracts unknown-frontier candidates:
   - `/exploration/frontier_candidates`
   - `/exploration/frontier_candidates_json`
7. `semantic_rl_decider_node` scores candidates and emits:
   - `/semantic_rl/candidate_scores_json`
   - `/semantic_rl/selected_goal`
8. `exploration_manager_node` sends `NavigateToPose` goals to Nav2.
9. Nav2 outputs `/cmd_vel`, safety layer filters to `/cmd_vel_safe`, then Ackermann adapter drives controller.

## RL Decision Features

- `distance_to_frontier`
- `estimated_free_gain`
- `heading_change`
- `local_obstacle_risk`
- `semantic_novelty_score`
- `semantic_priority_score`

## Offline RL Artifacts

- Dataset: `experiments/semantic_rl_dataset.jsonl`
- Trained policy: `experiments/semantic_rl_policy.json`

## Runtime Export

- Service: `/semantic_map/export`
- Output: `experiments/results/semantic_explorer/run_<timestamp>/semantic_map.json`
- Auto metrics: `experiments/results/semantic_explorer/run_<timestamp>/metrics.json`
