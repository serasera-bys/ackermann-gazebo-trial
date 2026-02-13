# Architecture (MVP)

## Data Flow

1. `hybrid_nav_rule_planner` reads `/odom` and generates nominal `Twist` on `/cmd_vel_raw`.
2. `hybrid_nav_safety_layer` reads `/cmd_vel_raw` (+ optional `/scan`) and publishes safe `Twist` on `/cmd_vel`.
3. Existing `ackermann_control_node` converts `/cmd_vel` to stamped reference for `ackermann_steering_controller`.
4. `hybrid_nav_metrics` logs odom/command/safety events for evaluation.

## Safety Behavior

- Always clamp linear/angular speed.
- If scan is available and minimum range is below stop threshold: force stop command.
- Count stop interventions for post-run analysis.

## Next Stage

- Replace rule planner with RL policy under the same `/cmd_vel_raw` interface.
- Add benchmark launcher for rule vs RL runs with identical scenarios.

