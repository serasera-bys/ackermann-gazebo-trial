# Results

This document should be updated after each benchmark run.

## Latest benchmark artifacts

- CSV: `experiments/results/benchmark_summary.csv`
- JSON: `experiments/results/benchmark_summary.json`
- Plot: `experiments/results/success_rate.png`
- Plot: `experiments/results/time_to_goal.png`
- Plot: `experiments/results/safety_hits.png`

## Result table template

| Scenario | Planner | Episodes | Success Rate | Mean Time to Goal (s) | Collision/Safety Rate | Mean Distance (m) |
|---|---:|---:|---:|---:|---:|---:|
| easy | rule | - | - | - | - | - |
| easy | rl | - | - | - | - | - |
| medium | rule | - | - | - | - | - |
| medium | rl | - | - | - | - | - |
| cluttered | rule | - | - | - | - | - |
| cluttered | rl | - | - | - | - | - |

## Engineering decisions

1. Offline RL instead of online training in runtime loop:
- Safer and easier to debug.
- Clear separation of data generation vs policy update.

2. Safety layer as hard guard:
- Prevents direct unsafe command pass-through.
- Adds explicit intervention signal for metrics.

3. Episode fallback strategy:
- Default fast reset (`model_only`).
- Escalation to full reset + controller respawn when rearm stalls.

## Notes for CV/demo

- Use a single benchmark run id and include the generated plot images in README.
- Mention the protocol (fixed seeds, fixed episode count) for reproducibility.
