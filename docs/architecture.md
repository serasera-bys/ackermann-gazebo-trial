# Architecture

## Runtime graph

1. Planner (`hybrid_nav_rule_planner` or `hybrid_nav_rl_planner`) publishes nominal command on `/cmd_vel_raw`.
2. `hybrid_nav_safety_layer` filters command using `/scan` and publishes safe command on `/cmd_vel`.
3. `ackermann_control_node` converts `/cmd_vel` into steering + wheel controller references.
4. `hybrid_episode_manager` closes the training loop:
   - detects terminal state (goal/stuck),
   - requests Gazebo reset,
   - performs fallback full reset when `model_only` re-arm fails,
   - respawns controllers after full reset,
   - publishes episode events on `/hybrid_nav/episode_event`.
5. `hybrid_nav_metrics` and `hybrid_rl_dataset_collector` subscribe episode events, so metrics and dataset are episode-scoped.

## Episode event contract

Events are JSON payloads on `/hybrid_nav/episode_event`:
- `episode_start`: includes `episode_id`, `goal_x`, `goal_y`, `goal_tolerance`
- `episode_end`: includes `episode_id`, `reason`, `dist_to_goal`
- `reset_requested` / `reset_failed`
- `episode_randomized` when domain randomization is enabled

Both planners also consume `episode_start` to update goal online (no node restart required).

## RL pipeline (offline, reward-based)

- Collector: records state/action transitions (`rl_dataset.jsonl`) from runtime.
- Trainer (`train_stub`): reward-weighted offline optimization from dataset.
  - reward components:
    - positive progress-to-goal,
    - goal bonus,
    - collision/safety penalty,
    - stuck-step penalty,
    - stuck-terminal penalty,
    - angular oscillation penalty.
- Policy artifact: `rl_policy.json` (gains + metadata), including:
  - `k_linear`, `k_heading`, `k_avoid`,
  - `k_front_brake` (obstacle-proximity brake term),
  - `k_heading_rate_limit` (heading slew limiter),
  - `training_stats`.
- RL planner: inference-only policy execution from saved policy file.
- Failure analysis helper: `experiments/analyze_rl_failures.py` on benchmark episode summaries.

This is offline RL-style training, not online learning inside Gazebo runtime loop.
