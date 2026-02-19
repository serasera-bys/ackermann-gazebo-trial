import argparse
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


def _default_experiments_dir() -> Path:
    env_value = os.environ.get("HYBRID_NAV_EXPERIMENTS_DIR", "").strip()
    if env_value:
        return Path(env_value)
    return Path.home() / ".ros" / "hybrid_nav_robot" / "experiments"


def parse_args() -> argparse.Namespace:
    default_dir = _default_experiments_dir()
    parser = argparse.ArgumentParser(
        description=(
            "Offline reward-weighted trainer for hybrid navigation policy gains. "
            "This optimizer is still lightweight, but now includes front-brake and "
            "heading slew-limit features for safer RL inference behavior."
        )
    )
    parser.add_argument(
        "--dataset",
        default=str(default_dir / "rl_dataset.jsonl"),
    )
    parser.add_argument(
        "--output",
        default=str(default_dir / "rl_policy.json"),
    )
    parser.add_argument("--max-linear", type=float, default=0.8)
    parser.add_argument("--max-angular", type=float, default=0.7)
    parser.add_argument("--avoid-distance", type=float, default=1.1)
    parser.add_argument("--min-fit-samples", type=int, default=80)
    parser.add_argument("--success-episode-weight", type=float, default=1.25)
    parser.add_argument("--balance-by-scenario", type=parse_bool, default=True)

    # Reward specification.
    parser.add_argument("--reward-progress-scale", type=float, default=6.0)
    parser.add_argument("--reward-goal-bonus", type=float, default=5.0)
    parser.add_argument("--penalty-stuck-terminal", type=float, default=3.0)
    parser.add_argument("--penalty-collision", type=float, default=2.0)
    parser.add_argument("--penalty-stuck-step", type=float, default=0.25)
    parser.add_argument("--penalty-angular-oscillation", type=float, default=0.08)
    parser.add_argument("--collision-threshold", type=float, default=0.24)
    parser.add_argument("--stuck-progress-eps", type=float, default=0.01)
    parser.add_argument("--stuck-cmd-threshold", type=float, default=0.10)

    # Reward -> regression weight mapping.
    parser.add_argument("--reward-weight-scale", type=float, default=0.7)
    parser.add_argument("--weight-min", type=float, default=0.10)
    parser.add_argument("--weight-max", type=float, default=8.0)

    # Data quality / regularization.
    parser.add_argument("--dedup-stall-progress-eps", type=float, default=0.0025)
    parser.add_argument("--dedup-stall-cmd-threshold", type=float, default=0.05)
    parser.add_argument("--l2-regularization", type=float, default=1e-3)
    parser.add_argument("--k-linear-min", type=float, default=0.40)
    parser.add_argument("--k-linear-max", type=float, default=2.5)
    parser.add_argument("--k-heading-min", type=float, default=0.75)
    parser.add_argument("--k-heading-max", type=float, default=3.5)
    parser.add_argument("--k-avoid-min", type=float, default=0.15)
    parser.add_argument("--k-front-brake-max", type=float, default=0.65)
    parser.add_argument("--k-avoid-max", type=float, default=2.5)
    parser.add_argument("--preset-label", type=str, default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    episodes = load_episodes(dataset_path)
    if not episodes:
        raise RuntimeError(f"No samples found in dataset: {dataset_path}")

    fit_candidates: list[dict[str, float | str]] = []
    reward_sum = 0.0
    reward_count = 0
    goal_episodes = 0
    stuck_episodes = 0
    collision_hits = 0
    removed_duplicates = 0
    heading_slew_samples: list[float] = []
    scenario_sample_counts: dict[str, int] = {}
    scenario_episode_counts: dict[str, int] = {}
    stall_heavy_episode_count = 0

    for _, episode_samples in episodes.items():
        if not episode_samples:
            continue

        filtered_samples, removed_count = dedup_stall_samples(episode_samples, args)
        removed_duplicates += removed_count
        if not filtered_samples:
            continue

        terminal_reason = find_terminal_reason(filtered_samples)
        success_episode = terminal_reason == "goal"
        if terminal_reason == "goal":
            goal_episodes += 1
        elif terminal_reason == "stuck":
            stuck_episodes += 1

        scenario_label = str(filtered_samples[0].get("scenario", "default"))
        scenario_episode_counts[scenario_label] = scenario_episode_counts.get(scenario_label, 0) + 1
        if is_stall_heavy_episode(filtered_samples, terminal_reason):
            stall_heavy_episode_count += 1

        for i, sample in enumerate(filtered_samples):
            next_sample = filtered_samples[i + 1] if i + 1 < len(filtered_samples) else None
            reward, collision_flag = compute_reward(sample, next_sample, args)
            reward_sum += reward
            reward_count += 1
            collision_hits += int(collision_flag)

            if sample["done"]:
                continue

            base_weight = clamp(
                1.0 + (args.reward_weight_scale * reward),
                args.weight_min,
                args.weight_max,
            )
            if success_episode:
                base_weight *= args.success_episode_weight

            fit_candidates.append(
                {
                    "scenario": scenario_label,
                    "dist": sample["dist"],
                    "heading_error": sample["heading_error"],
                    "avoid_feature": avoid_feature(
                        sample["left_range"],
                        sample["right_range"],
                        args.avoid_distance,
                    ),
                    "front_brake": front_brake_feature(
                        sample["front_range"],
                        args.avoid_distance,
                    ),
                    "action_linear": sample["action_linear"],
                    "action_angular": sample["action_angular"],
                    "weight": base_weight,
                }
            )
            scenario_sample_counts[scenario_label] = (
                scenario_sample_counts.get(scenario_label, 0) + 1
            )

            if next_sample is not None:
                dt = max(1e-3, next_sample["episode_time_sec"] - sample["episode_time_sec"])
                if 0.0 < dt <= 0.5:
                    heading_slew_samples.append(
                        abs(next_sample["action_angular"] - sample["action_angular"]) / dt
                    )

    if len(fit_candidates) < args.min_fit_samples:
        raise RuntimeError(
            f"Not enough non-terminal samples for fit ({len(fit_candidates)}), "
            f"need at least {args.min_fit_samples}"
        )

    scenario_weights = compute_scenario_weights(scenario_sample_counts, args.balance_by_scenario)
    fit_samples = apply_scenario_balance(fit_candidates, scenario_weights, args)

    k_linear, k_front_brake, k_heading, k_avoid = fit_weighted_policy(
        fit_samples, args
    )
    k_heading_rate_limit = estimate_heading_rate_limit(heading_slew_samples)

    reward_mean = (reward_sum / reward_count) if reward_count > 0 else 0.0
    policy = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "training_method": "reward_weighted_regression_v2",
        "dataset_path": str(dataset_path),
        "preset_label": str(args.preset_label).strip(),
        "episode_count": len(episodes),
        "fit_sample_count": len(fit_samples),
        "reward_mean": reward_mean,
        "goal_episodes": goal_episodes,
        "stuck_episodes": stuck_episodes,
        "collision_penalty_hits": collision_hits,
        "k_linear": k_linear,
        "k_heading": k_heading,
        "k_avoid": k_avoid,
        "k_front_brake": k_front_brake,
        "k_heading_rate_limit": k_heading_rate_limit,
        "max_linear_speed": args.max_linear,
        "max_angular_speed": args.max_angular,
        "avoid_distance": args.avoid_distance,
        "reward_spec": {
            "progress_scale": args.reward_progress_scale,
            "goal_bonus": args.reward_goal_bonus,
            "stuck_terminal_penalty": args.penalty_stuck_terminal,
            "collision_penalty": args.penalty_collision,
            "stuck_step_penalty": args.penalty_stuck_step,
            "angular_oscillation_penalty": args.penalty_angular_oscillation,
            "collision_threshold": args.collision_threshold,
            "stuck_progress_eps": args.stuck_progress_eps,
            "stuck_cmd_threshold": args.stuck_cmd_threshold,
            "reward_weight_scale": args.reward_weight_scale,
            "weight_min": args.weight_min,
            "weight_max": args.weight_max,
            "success_episode_weight": args.success_episode_weight,
            "balance_by_scenario": args.balance_by_scenario,
            "l2_regularization": args.l2_regularization,
        },
        "training_stats": {
            "removed_stall_duplicates": removed_duplicates,
            "scenario_sample_counts": scenario_sample_counts,
            "scenario_episode_counts": scenario_episode_counts,
            "scenario_weight_factors": scenario_weights,
            "heading_slew_sample_count": len(heading_slew_samples),
            "heading_slew_p90": percentile(heading_slew_samples, 90.0),
            "reward_mean": reward_mean,
            "goal_rate": clamp(goal_episodes / max(len(episodes), 1), 0.0, 1.0),
            "stall_heavy_episode_count": stall_heavy_episode_count,
            "stall_heavy_episode_rate": clamp(
                stall_heavy_episode_count / max(len(episodes), 1), 0.0, 1.0
            ),
        },
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(policy, indent=2), encoding="utf-8")
    print(f"Saved policy to {out}")
    print(json.dumps(policy, indent=2))


def load_episodes(path: Path) -> dict[str, list[dict[str, Any]]]:
    # Dataset can contain multiple collection sessions where episode_id restarts from 1.
    # We split by scenario+episode_id and also segment when episode_time goes backwards.
    episodes: dict[str, list[dict[str, Any]]] = {}
    last_time_by_pair: dict[tuple[str, int], float] = {}
    segment_by_pair: dict[tuple[str, int], int] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            sample = {
                "episode_id": int(raw.get("episode_id", 0)),
                "episode_time_sec": float(raw.get("episode_time_sec", 0.0)),
                "scenario": str(raw.get("scenario", "default")),
                "dist": float(raw.get("dist", 0.0)),
                "heading_error": float(raw.get("heading_error", 0.0)),
                "front_range": float(raw.get("front_range", -1.0)),
                "left_range": float(raw.get("left_range", -1.0)),
                "right_range": float(raw.get("right_range", -1.0)),
                "action_linear": float(raw.get("action_linear", 0.0)),
                "action_angular": float(raw.get("action_angular", 0.0)),
                "done": bool(raw.get("done", False)),
                "terminal_reason": str(raw.get("terminal_reason", "")),
            }

            pair = (sample["scenario"], sample["episode_id"])
            segment = segment_by_pair.get(pair, 0)
            prev_t = last_time_by_pair.get(pair)
            cur_t = float(sample["episode_time_sec"])
            if prev_t is not None and (cur_t + 0.2) < prev_t:
                segment += 1
                segment_by_pair[pair] = segment
            key = f"{sample['scenario']}|{sample['episode_id']}|{segment}"
            episodes.setdefault(key, []).append(sample)
            last_time_by_pair[pair] = cur_t

    for sample_list in episodes.values():
        sample_list.sort(key=lambda s: float(s.get("episode_time_sec", 0.0)))
    return episodes


def is_stall_heavy_episode(samples: list[dict[str, Any]], terminal_reason: str) -> bool:
    if not samples:
        return False
    if terminal_reason == "stuck":
        return True
    start_dist = float(samples[0].get("dist", 0.0))
    min_dist = min(float(s.get("dist", start_dist)) for s in samples)
    progress = max(0.0, start_dist - min_dist)
    episode_time = float(samples[-1].get("episode_time_sec", 0.0))
    return progress < 0.7 and episode_time >= 6.0


def dedup_stall_samples(
    samples: list[dict[str, Any]],
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], int]:
    filtered: list[dict[str, Any]] = []
    removed = 0
    for sample in samples:
        if not filtered:
            filtered.append(sample)
            continue
        prev = filtered[-1]
        if sample["done"]:
            filtered.append(sample)
            continue
        if prev["done"]:
            filtered.append(sample)
            continue

        tiny_progress = abs(sample["dist"] - prev["dist"]) <= args.dedup_stall_progress_eps
        low_cmd = (
            abs(sample["action_linear"]) <= args.dedup_stall_cmd_threshold
            and abs(prev["action_linear"]) <= args.dedup_stall_cmd_threshold
        )
        almost_same_turn = abs(sample["action_angular"] - prev["action_angular"]) <= 0.03
        almost_same_front = abs(sample["front_range"] - prev["front_range"]) <= 0.03

        if tiny_progress and low_cmd and almost_same_turn and almost_same_front:
            removed += 1
            continue
        filtered.append(sample)
    return filtered, removed


def find_terminal_reason(samples: list[dict[str, Any]]) -> str:
    for sample in reversed(samples):
        reason = str(sample.get("terminal_reason", "")).strip().lower()
        if reason in ("goal", "stuck"):
            return reason
    return ""


def compute_reward(
    sample: dict[str, Any],
    next_sample: Optional[dict[str, Any]],
    args: argparse.Namespace,
) -> tuple[float, bool]:
    next_dist = sample["dist"] if next_sample is None else next_sample["dist"]
    progress = sample["dist"] - next_dist
    reward = args.reward_progress_scale * progress

    min_range = finite_min([sample["front_range"], sample["left_range"], sample["right_range"]])
    collision_flag = min_range is not None and min_range < args.collision_threshold
    if collision_flag:
        reward -= args.penalty_collision

    moving_cmd = abs(sample["action_linear"]) >= args.stuck_cmd_threshold
    tiny_progress = abs(progress) < args.stuck_progress_eps
    if moving_cmd and tiny_progress:
        reward -= args.penalty_stuck_step

    if next_sample is not None:
        reward -= args.penalty_angular_oscillation * abs(
            next_sample["action_angular"] - sample["action_angular"]
        )

    if sample["done"] and sample["terminal_reason"] == "goal":
        reward += args.reward_goal_bonus
    if sample["done"] and sample["terminal_reason"] == "stuck":
        reward -= args.penalty_stuck_terminal

    return reward, collision_flag


def compute_scenario_weights(
    scenario_sample_counts: dict[str, int], balance_by_scenario: bool
) -> dict[str, float]:
    if not scenario_sample_counts:
        return {"default": 1.0}
    if not balance_by_scenario:
        return {k: 1.0 for k in scenario_sample_counts}

    total = float(sum(scenario_sample_counts.values()))
    scenario_count = float(len(scenario_sample_counts))
    factors: dict[str, float] = {}
    for scenario, count in scenario_sample_counts.items():
        raw = total / max(1.0, scenario_count * float(count))
        factors[scenario] = clamp(raw, 0.5, 2.0)
    return factors


def apply_scenario_balance(
    fit_candidates: list[dict[str, float | str]],
    scenario_weights: dict[str, float],
    args: argparse.Namespace,
) -> list[dict[str, float]]:
    samples: list[dict[str, float]] = []
    for row in fit_candidates:
        scenario = str(row["scenario"])
        weight = float(row["weight"]) * float(scenario_weights.get(scenario, 1.0))
        weight = clamp(weight, args.weight_min, args.weight_max)
        samples.append(
            {
                "dist": float(row["dist"]),
                "heading_error": float(row["heading_error"]),
                "avoid_feature": float(row["avoid_feature"]),
                "front_brake": float(row["front_brake"]),
                "action_linear": float(row["action_linear"]),
                "action_angular": float(row["action_angular"]),
                "weight": weight,
            }
        )
    return samples


def fit_weighted_policy(
    samples: list[dict[str, float]],
    args: argparse.Namespace,
) -> tuple[float, float, float, float]:
    # Linear model:
    #   linear ~ k_linear * dist - k_front_brake * front_brake
    # Solved as [a, b] where a=k_linear and b multiplies front_brake directly.
    l2_reg = float(args.l2_regularization)
    s_xx = l2_reg
    s_xy = 0.0
    s_yy = l2_reg
    s_xt = 0.0
    s_yt = 0.0

    # Angular model:
    #   angular ~ k_heading * heading_error + k_avoid * avoid_feature
    s_hh = l2_reg
    s_ha = 0.0
    s_aa = l2_reg
    s_hy = 0.0
    s_ay = 0.0

    for s in samples:
        w = s["weight"]
        d = s["dist"]
        fb = s["front_brake"]
        h = s["heading_error"]
        a = s["avoid_feature"]
        lin = s["action_linear"]
        ang = s["action_angular"]

        s_xx += w * d * d
        s_xy += w * d * fb
        s_yy += w * fb * fb
        s_xt += w * d * lin
        s_yt += w * fb * lin

        s_hh += w * h * h
        s_ha += w * h * a
        s_aa += w * a * a
        s_hy += w * h * ang
        s_ay += w * a * ang

    linear_det = s_xx * s_yy - s_xy * s_xy
    if abs(linear_det) > 1e-9:
        a = (s_xt * s_yy - s_yt * s_xy) / linear_det
        b = (s_yt * s_xx - s_xt * s_xy) / linear_det
    else:
        a = 0.8
        b = -0.8

    angular_det = s_hh * s_aa - s_ha * s_ha
    if abs(angular_det) > 1e-9:
        k_heading = (s_hy * s_aa - s_ay * s_ha) / angular_det
        k_avoid = (s_ay * s_hh - s_hy * s_ha) / angular_det
    else:
        k_heading = 1.2
        k_avoid = 0.6

    # Keep a non-trivial forward gain to avoid near-zero linear command collapse.
    k_linear = clamp(a, args.k_linear_min, args.k_linear_max)
    k_front_brake = clamp(-b, 0.0, args.k_front_brake_max)
    k_heading = clamp(k_heading, args.k_heading_min, args.k_heading_max)
    # avoid_feature is (left - right), so positive gain steers toward open space.
    # Negative gain would steer toward closer obstacles and is unsafe.
    k_avoid = clamp(k_avoid, args.k_avoid_min, args.k_avoid_max)
    return k_linear, k_front_brake, k_heading, k_avoid


def avoid_feature(left: float, right: float, avoid_distance: float) -> float:
    if left < 0.0:
        left = avoid_distance
    if right < 0.0:
        right = avoid_distance
    return clamp((left - right) / max(avoid_distance, 1e-6), -1.0, 1.0)


def front_brake_feature(front: float, avoid_distance: float) -> float:
    if not math.isfinite(front) or front < 0.0:
        return 0.0
    return clamp((avoid_distance - front) / max(avoid_distance, 1e-6), 0.0, 1.0)


def estimate_heading_rate_limit(heading_slew_samples: list[float]) -> float:
    if not heading_slew_samples:
        return 1.2
    p90 = percentile(heading_slew_samples, 90.0)
    return clamp(1.2 * p90, 1.0, 4.0)


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = int(round((p / 100.0) * (len(ordered) - 1)))
    idx = max(0, min(len(ordered) - 1, idx))
    return ordered[idx]


def finite_min(values: list[float]) -> Optional[float]:
    finite_values = [v for v in values if math.isfinite(v) and v >= 0.0]
    if not finite_values:
        return None
    return min(finite_values)


def parse_bool(value: str) -> bool:
    return str(value).strip().lower() in ("1", "true", "yes", "y", "on")


def clamp(value: float, min_v: float, max_v: float) -> float:
    return max(min_v, min(max_v, value))


if __name__ == "__main__":
    main()
