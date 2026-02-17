import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Offline reward-weighted trainer for hybrid navigation policy gains. "
            "This is a lightweight reward-based optimizer over logged dataset samples."
        )
    )
    parser.add_argument(
        "--dataset",
        default="/home/bernard/ros2_ws/src/hybrid_nav_robot/experiments/rl_dataset.jsonl",
    )
    parser.add_argument(
        "--output",
        default="/home/bernard/ros2_ws/src/hybrid_nav_robot/experiments/rl_policy.json",
    )
    parser.add_argument("--max-linear", type=float, default=0.8)
    parser.add_argument("--max-angular", type=float, default=0.7)
    parser.add_argument("--avoid-distance", type=float, default=1.1)
    parser.add_argument("--min-fit-samples", type=int, default=50)

    # Reward specification.
    parser.add_argument("--reward-progress-scale", type=float, default=6.0)
    parser.add_argument("--reward-goal-bonus", type=float, default=5.0)
    parser.add_argument("--penalty-stuck-terminal", type=float, default=3.0)
    parser.add_argument("--penalty-collision", type=float, default=2.0)
    parser.add_argument("--penalty-stuck-step", type=float, default=0.25)
    parser.add_argument("--collision-threshold", type=float, default=0.24)
    parser.add_argument("--stuck-progress-eps", type=float, default=0.01)
    parser.add_argument("--stuck-cmd-threshold", type=float, default=0.10)

    # Reward -> regression weight mapping.
    parser.add_argument("--reward-weight-scale", type=float, default=0.7)
    parser.add_argument("--weight-min", type=float, default=0.10)
    parser.add_argument("--weight-max", type=float, default=8.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    episodes = load_episodes(dataset_path)
    if not episodes:
        raise RuntimeError(f"No samples found in dataset: {dataset_path}")

    fit_samples: list[dict[str, float]] = []
    reward_sum = 0.0
    reward_count = 0
    goal_episodes = 0
    stuck_episodes = 0
    collision_hits = 0

    for _, episode_samples in episodes.items():
        for i, sample in enumerate(episode_samples):
            next_dist = sample["dist"]
            if i + 1 < len(episode_samples):
                next_dist = episode_samples[i + 1]["dist"]

            reward, collision_flag = compute_reward(sample, next_dist, args)
            reward_sum += reward
            reward_count += 1
            collision_hits += int(collision_flag)

            if sample["done"]:
                if sample["terminal_reason"] == "goal":
                    goal_episodes += 1
                elif sample["terminal_reason"] == "stuck":
                    stuck_episodes += 1

            if sample["done"]:
                continue

            weight = clamp(
                1.0 + (args.reward_weight_scale * reward),
                args.weight_min,
                args.weight_max,
            )
            fit_samples.append(
                {
                    "dist": sample["dist"],
                    "heading_error": sample["heading_error"],
                    "avoid_feature": avoid_feature(sample["left_range"], sample["right_range"], args.avoid_distance),
                    "action_linear": sample["action_linear"],
                    "action_angular": sample["action_angular"],
                    "weight": weight,
                }
            )

    if len(fit_samples) < args.min_fit_samples:
        raise RuntimeError(
            f"Not enough non-terminal samples for fit ({len(fit_samples)}), "
            f"need at least {args.min_fit_samples}"
        )

    k_linear, k_heading, k_avoid = fit_weighted_policy(fit_samples)
    policy = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "training_method": "reward_weighted_regression",
        "dataset_path": str(dataset_path),
        "episode_count": len(episodes),
        "fit_sample_count": len(fit_samples),
        "reward_mean": (reward_sum / reward_count) if reward_count > 0 else 0.0,
        "goal_episodes": goal_episodes,
        "stuck_episodes": stuck_episodes,
        "collision_penalty_hits": collision_hits,
        "k_linear": k_linear,
        "k_heading": k_heading,
        "k_avoid": k_avoid,
        "max_linear_speed": args.max_linear,
        "max_angular_speed": args.max_angular,
        "avoid_distance": args.avoid_distance,
        "reward_spec": {
            "progress_scale": args.reward_progress_scale,
            "goal_bonus": args.reward_goal_bonus,
            "stuck_terminal_penalty": args.penalty_stuck_terminal,
            "collision_penalty": args.penalty_collision,
            "stuck_step_penalty": args.penalty_stuck_step,
            "collision_threshold": args.collision_threshold,
            "stuck_progress_eps": args.stuck_progress_eps,
            "stuck_cmd_threshold": args.stuck_cmd_threshold,
            "reward_weight_scale": args.reward_weight_scale,
            "weight_min": args.weight_min,
            "weight_max": args.weight_max,
        },
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(policy, indent=2), encoding="utf-8")
    print(f"Saved policy to {out}")
    print(json.dumps(policy, indent=2))


def load_episodes(path: Path) -> dict[int, list[dict[str, Any]]]:
    episodes: dict[int, list[dict[str, Any]]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            sample = {
                "episode_id": int(raw.get("episode_id", 0)),
                "dist": float(raw.get("dist", 0.0)),
                "heading_error": float(raw.get("heading_error", 0.0)),
                "left_range": float(raw.get("left_range", -1.0)),
                "right_range": float(raw.get("right_range", -1.0)),
                "front_range": float(raw.get("front_range", -1.0)),
                "action_linear": float(raw.get("action_linear", 0.0)),
                "action_angular": float(raw.get("action_angular", 0.0)),
                "done": bool(raw.get("done", False)),
                "terminal_reason": str(raw.get("terminal_reason", "")),
            }
            episodes.setdefault(sample["episode_id"], []).append(sample)
    return episodes


def compute_reward(
    sample: dict[str, Any], next_dist: float, args: argparse.Namespace
) -> tuple[float, bool]:
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

    if sample["done"] and sample["terminal_reason"] == "goal":
        reward += args.reward_goal_bonus
    if sample["done"] and sample["terminal_reason"] == "stuck":
        reward -= args.penalty_stuck_terminal

    return reward, collision_flag


def fit_weighted_policy(samples: list[dict[str, float]]) -> tuple[float, float, float]:
    lin_num = 0.0
    lin_den = 0.0

    s_hh = 0.0
    s_ha = 0.0
    s_aa = 0.0
    s_hy = 0.0
    s_ay = 0.0

    for s in samples:
        w = s["weight"]
        d = s["dist"]
        h = s["heading_error"]
        a = s["avoid_feature"]
        lin = s["action_linear"]
        ang = s["action_angular"]

        lin_num += w * d * lin
        lin_den += w * d * d
        s_hh += w * h * h
        s_ha += w * h * a
        s_aa += w * a * a
        s_hy += w * h * ang
        s_ay += w * a * ang

    k_linear = lin_num / lin_den if lin_den > 1e-9 else 0.8
    det = s_hh * s_aa - s_ha * s_ha
    if abs(det) > 1e-9:
        k_heading = (s_hy * s_aa - s_ay * s_ha) / det
        k_avoid = (s_ay * s_hh - s_hy * s_ha) / det
    else:
        k_heading = 1.2
        k_avoid = 0.6

    k_linear = clamp(k_linear, 0.1, 2.5)
    k_heading = clamp(k_heading, 0.1, 3.5)
    k_avoid = clamp(k_avoid, -2.5, 2.5)
    return k_linear, k_heading, k_avoid


def avoid_feature(left: float, right: float, avoid_distance: float) -> float:
    if left < 0.0:
        left = avoid_distance
    if right < 0.0:
        right = avoid_distance
    return clamp((left - right) / max(avoid_distance, 1e-6), -1.0, 1.0)


def finite_min(values: list[float]) -> Optional[float]:
    finite_values = [v for v in values if math.isfinite(v) and v >= 0.0]
    if not finite_values:
        return None
    return min(finite_values)


def clamp(value: float, min_v: float, max_v: float) -> float:
    return max(min_v, min(max_v, value))


if __name__ == "__main__":
    main()
