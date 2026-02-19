# Results

This document should be updated after each benchmark run.

## Latest benchmark artifacts

- CSV: `experiments/results/benchmark_summary.csv`
- JSON: `experiments/results/benchmark_summary.json`
- Plot: `experiments/results/success_rate.png`
- Plot: `experiments/results/time_to_goal.png`
- Plot: `experiments/results/safety_hits.png`
- Source run id: `run_20260218_213347` (120 episodes total, 20 per scenario/planner)
- Locked baseline snapshot: `docs/baseline_pre_uplift_20260217_221631.json`

## Final benchmark (post-uplift)

| Scenario | Planner | Episodes | Success Rate | Mean Time to Goal (s) | Collision/Safety Rate | Mean Distance (m) | Mean Safety Hits |
|---|---:|---:|---:|---:|---:|---:|---:|
| easy | rule | 20 | 1.0000 | 22.9328 | 0.7000 | 12.4499 | 54.55 |
| easy | rl | 20 | 0.4500 | 8.4129 | 0.6500 | 3.5916 | 51.65 |
| medium | rule | 20 | 1.0000 | 18.0256 | 0.5000 | 10.8102 | 20.10 |
| medium | rl | 20 | 0.6500 | 10.6154 | 0.6000 | 3.5336 | 96.40 |
| cluttered | rule | 20 | 0.9000 | 22.3666 | 0.7500 | 11.8806 | 40.60 |
| cluttered | rl | 20 | 0.6500 | 8.9066 | 0.6000 | 3.8543 | 62.00 |

Target check:
- `easy` RL success_rate >= 0.50: `0.45` (not met)
- `medium` RL success_rate >= 0.50: `0.65` (met)
- `cluttered` RL success_rate >= 0.30: `0.65` (met)
- benchmark matrix complete: `120/120 episodes` (met)

## Baseline (pre-uplift)

| Scenario | Planner | Episodes | Success Rate | Mean Time to Goal (s) | Collision/Safety Rate | Mean Distance (m) |
|---|---:|---:|---:|---:|---:|---:|
| easy | rule | 3 | 1.000000 | 22.1860 | 0.666667 | 10.8172 |
| easy | rl | 3 | 0.000000 | N/A | 0.333333 | 0.0471 |
| medium | rule | 3 | 1.000000 | 21.1869 | 0.666667 | 11.3115 |
| medium | rl | 3 | 0.333333 | 17.8006 | 0.333333 | 2.0463 |
| cluttered | rule | 3 | 1.000000 | 25.4612 | 1.000000 | 14.7724 |
| cluttered | rl | 3 | 0.000000 | N/A | 0.666667 | 1.4221 |

RL failure signature in this baseline:
- `easy`: `3/3` ended `stuck`
- `cluttered`: `3/3` ended `stuck`
- `medium`: `2/3` ended `stuck`, `1/3` `goal`

## RL uplift targets

- easy RL success_rate >= 0.50
- medium RL success_rate >= 0.50
- cluttered RL success_rate >= 0.30
- benchmark matrix complete (20 episodes per scenario/planner) with no incomplete runs

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
- Run `experiments/analyze_rl_failures.py` on RL episodes to explain failure modes and tuning decisions.
