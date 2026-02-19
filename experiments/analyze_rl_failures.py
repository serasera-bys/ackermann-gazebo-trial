#!/usr/bin/env python3
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze RL failure patterns from benchmark episode summary jsonl files."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to a run raw directory or a single episodes_*.jsonl file",
    )
    parser.add_argument(
        "--planner-mode",
        default="rl",
        help="Planner mode filter (default: rl)",
    )
    parser.add_argument(
        "--safety-heavy-threshold",
        type=int,
        default=40,
        help="Episode is safety-heavy if safety_interventions >= threshold",
    )
    parser.add_argument(
        "--low-progress-threshold",
        type=float,
        default=2.0,
        help="Episode is low-progress if distance_traveled_m < threshold",
    )
    parser.add_argument(
        "--near-collision-threshold",
        type=float,
        default=0.30,
        help="Episode is near-collision if min_obstacle_range_m <= threshold",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Optional output JSON path for analysis report",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    episode_files = resolve_episode_files(input_path)
    if not episode_files:
        raise RuntimeError(f"No episode summary files found for input: {input_path}")

    episodes = load_episodes(episode_files, planner_mode=args.planner_mode)
    if not episodes:
        raise RuntimeError(
            f"No episodes found for planner_mode='{args.planner_mode}' in {input_path}"
        )

    report = build_report(
        episodes=episodes,
        planner_mode=args.planner_mode,
        safety_heavy_threshold=args.safety_heavy_threshold,
        low_progress_threshold=args.low_progress_threshold,
        near_collision_threshold=args.near_collision_threshold,
    )

    output_path = Path(args.output) if args.output else default_output_path(input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Wrote failure analysis: {output_path}")
    print(format_summary(report))


def resolve_episode_files(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    raw_dir = input_path
    if input_path.name.startswith("run_"):
        raw_dir = input_path / "raw"
    return sorted(raw_dir.glob("episodes_*.jsonl"))


def load_episodes(files: list[Path], planner_mode: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in files:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if str(row.get("planner_mode", "")).strip().lower() != planner_mode:
                    continue
                reason = str(row.get("reason", "")).strip().lower()
                if reason not in ("goal", "stuck"):
                    continue
                records.append(row)
    return records


def build_report(
    episodes: list[dict[str, Any]],
    planner_mode: str,
    safety_heavy_threshold: int,
    low_progress_threshold: float,
    near_collision_threshold: float,
) -> dict[str, Any]:
    total = len(episodes)
    failed = [ep for ep in episodes if str(ep.get("reason", "")) != "goal"]
    success = total - len(failed)

    reason_counts = Counter(str(ep.get("reason", "unknown")) for ep in episodes)
    scenario_counts = Counter(str(ep.get("scenario", "default")) for ep in episodes)
    fail_by_scenario = Counter(str(ep.get("scenario", "default")) for ep in failed)

    safety_heavy = [
        ep for ep in failed if int(ep.get("safety_interventions", 0)) >= safety_heavy_threshold
    ]
    low_progress = [
        ep for ep in failed if float(ep.get("distance_traveled_m", 0.0)) < low_progress_threshold
    ]
    near_collision = [
        ep
        for ep in failed
        if float(ep.get("min_obstacle_range_m", -1.0)) >= 0.0
        and float(ep.get("min_obstacle_range_m", -1.0)) <= near_collision_threshold
    ]
    oscillation_proxy = [
        ep
        for ep in failed
        if int(ep.get("safety_interventions", 0)) >= safety_heavy_threshold
        and float(ep.get("distance_traveled_m", 0.0)) < low_progress_threshold
    ]

    top_safety = sorted(
        failed,
        key=lambda ep: int(ep.get("safety_interventions", 0)),
        reverse=True,
    )[:5]

    scenario_stats: dict[str, dict[str, float]] = defaultdict(dict)
    for scenario in scenario_counts:
        scoped = [ep for ep in episodes if str(ep.get("scenario", "default")) == scenario]
        scoped_fail = [ep for ep in scoped if str(ep.get("reason", "")) != "goal"]
        scenario_stats[scenario] = {
            "episode_count": float(len(scoped)),
            "success_rate": safe_ratio(len(scoped) - len(scoped_fail), len(scoped)),
            "mean_runtime_sec": safe_mean([float(ep.get("runtime_sec", 0.0)) for ep in scoped]),
            "mean_distance_m": safe_mean(
                [float(ep.get("distance_traveled_m", 0.0)) for ep in scoped]
            ),
            "mean_safety_hits": safe_mean(
                [float(ep.get("safety_interventions", 0.0)) for ep in scoped]
            ),
        }

    return {
        "planner_mode": planner_mode,
        "episode_count": total,
        "success_count": success,
        "success_rate": safe_ratio(success, total),
        "reason_counts": dict(reason_counts),
        "scenario_counts": dict(scenario_counts),
        "failure_count": len(failed),
        "failure_ratio": safe_ratio(len(failed), total),
        "failure_hotspots": {
            "by_scenario": dict(fail_by_scenario),
            "safety_heavy_count": len(safety_heavy),
            "low_progress_count": len(low_progress),
            "near_collision_count": len(near_collision),
            "oscillation_proxy_count": len(oscillation_proxy),
        },
        "failed_episode_stats": {
            "mean_runtime_sec": safe_mean([float(ep.get("runtime_sec", 0.0)) for ep in failed]),
            "mean_distance_m": safe_mean(
                [float(ep.get("distance_traveled_m", 0.0)) for ep in failed]
            ),
            "mean_goal_distance_m": safe_mean(
                [float(ep.get("goal_distance_m", -1.0)) for ep in failed if ep.get("goal_distance_m", -1.0) >= 0.0]
            ),
            "mean_safety_hits": safe_mean(
                [float(ep.get("safety_interventions", 0.0)) for ep in failed]
            ),
            "mean_min_obstacle_range_m": safe_mean(
                [float(ep.get("min_obstacle_range_m", -1.0)) for ep in failed if ep.get("min_obstacle_range_m", -1.0) >= 0.0]
            ),
        },
        "scenario_stats": scenario_stats,
        "top_safety_failed_episodes": [
            {
                "scenario": ep.get("scenario", "default"),
                "episode_id": ep.get("episode_id", -1),
                "reason": ep.get("reason", "unknown"),
                "runtime_sec": ep.get("runtime_sec", 0.0),
                "distance_traveled_m": ep.get("distance_traveled_m", 0.0),
                "goal_distance_m": ep.get("goal_distance_m", -1.0),
                "safety_interventions": ep.get("safety_interventions", 0),
                "min_obstacle_range_m": ep.get("min_obstacle_range_m", -1.0),
            }
            for ep in top_safety
        ],
        "thresholds": {
            "safety_heavy_threshold": safety_heavy_threshold,
            "low_progress_threshold": low_progress_threshold,
            "near_collision_threshold": near_collision_threshold,
        },
    }


def safe_ratio(num: int, den: int) -> float:
    return float(num) / float(den) if den > 0 else 0.0


def safe_mean(values: list[float]) -> float:
    return fmean(values) if values else 0.0


def default_output_path(input_path: Path) -> Path:
    if input_path.is_file():
        return input_path.with_name("rl_failure_analysis.json")
    if input_path.name.startswith("run_"):
        return input_path / "rl_failure_analysis.json"
    return input_path / "rl_failure_analysis.json"


def format_summary(report: dict[str, Any]) -> str:
    hotspots = report["failure_hotspots"]
    return (
        f"planner={report['planner_mode']}, episodes={report['episode_count']}, "
        f"success_rate={report['success_rate']:.3f}, failures={report['failure_count']}\n"
        f"hotspots: safety_heavy={hotspots['safety_heavy_count']}, "
        f"low_progress={hotspots['low_progress_count']}, "
        f"near_collision={hotspots['near_collision_count']}, "
        f"oscillation_proxy={hotspots['oscillation_proxy_count']}"
    )


if __name__ == "__main__":
    main()
