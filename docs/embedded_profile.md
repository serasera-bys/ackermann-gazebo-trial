# Embedded-Style System Profile

This project is not bare-metal firmware, but it follows embedded-style robotics engineering:
- bounded loop rates
- explicit timeout handling
- reset/recovery state machines
- deterministic safety guards

## Timing budget and control rates

- Planner loop (`rule` / `rl`): default `20 Hz`
- Safety layer: callback-driven guard on `/scan` + `/cmd_vel_raw`
- ros2_control update period: `0.02 s` (50 Hz target from controller config)
- Metrics heartbeat: `2.0 s`
- Episode manager checks:
  - controller active check period: `0.5 s`
  - stuck timeout: `5.0 s` (configurable)
  - episode max duration timeout: `60.0 s` in benchmark mode
  - command stale timeout: `1.0 s`

## Failure handling and recovery path

1. Planner publishes nominal command to `/cmd_vel_raw`.
2. Safety layer clamps/stops and publishes `/cmd_vel`.
3. Episode manager detects terminal states (`goal` / `stuck`).
4. Primary reset mode: `model_only` for fast loop.
5. If re-arm fails within timeout:
   - fallback full reset (`all`)
   - controller respawn (`joint_state_broadcaster`, `ackermann_steering_controller`, `front_wheel_velocity_controller`)
6. Episode restarts only after controller readiness gate.

## Noise and robustness

- Laser scan consumption uses sensor-data QoS profile.
- Safety intervention topic contributes to metrics and failure analysis.
- RL trainer penalizes:
  - collision proximity
  - no-progress while commanded forward
  - angular oscillation (new uplift penalty).

## Why this maps to embedded/system skill

- Real-time-ish rate-limited loops and deterministic watchdog-like recovery.
- Explicit degraded-mode path (`model_only` -> `all` reset fallback).
- Clear telemetry artifacts (`latest_metrics.json`, episode summaries, benchmark run logs).
