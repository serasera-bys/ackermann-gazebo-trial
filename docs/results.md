# Results

This document should be updated after each benchmark run.

## Latest benchmark artifacts

- CSV: `experiments/results/benchmark_summary.csv`
- JSON: `experiments/results/benchmark_summary.json`
- Plot: `experiments/results/success_rate.png`
- Plot: `experiments/results/time_to_goal.png`
- Plot: `experiments/results/safety_hits.png`
- Source run id: `run_20260216_232901` (18 episodes total, 3 per scenario/planner)

## Result table template

| Scenario | Planner | Episodes | Success Rate | Mean Time to Goal (s) | Collision/Safety Rate | Mean Distance (m) |
|---|---:|---:|---:|---:|---:|---:|
| easy | rule | 3 | 1.000000 | 38.2036 | 1.000000 | 17.6943 |
| easy | rl | 3 | 0.000000 | N/A | 1.000000 | 0.3831 |
| medium | rule | 3 | 1.000000 | 28.4661 | 1.000000 | 14.5574 |
| medium | rl | 3 | 0.333333 | 38.2225 | 0.666667 | 3.7614 |
| cluttered | rule | 3 | 1.000000 | 25.5759 | 0.666667 | 14.3750 |
| cluttered | rl | 3 | 0.000000 | N/A | 0.666667 | 1.4995 |

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
