#!/usr/bin/env python3
import argparse
import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable


@dataclass
class EpisodeRecord:
    scenario: str
    planner_mode: str
    reason: str
    goal_reached: bool
    runtime_sec: float
    distance_traveled_m: float
    goal_distance_m: float
    safety_interventions: int
    collision_flag: bool
    min_obstacle_range_m: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize benchmark episode logs.")
    parser.add_argument("--input-dir", required=True, help="Directory with episodes_*.jsonl files")
    parser.add_argument("--output-dir", required=True, help="Directory for canonical summary artifacts")
    parser.add_argument("--run-dir", required=True, help="Run-specific output directory")
    parser.add_argument("--run-id", default="", help="Optional benchmark run id/timestamp")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    run_dir = Path(args.run_dir)
    run_id = args.run_id or run_dir.name

    records = load_records(input_dir)
    if not records:
        raise RuntimeError(f"No episode records found in {input_dir}")

    rows = aggregate(records)
    metadata = {
        "run_id": run_id,
        "input_dir": str(input_dir),
        "total_episode_count": len(records),
        "group_count": len(rows),
    }
    summary = {"metadata": metadata, "rows": rows}

    run_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_json = run_dir / "benchmark_summary.json"
    run_csv = run_dir / "benchmark_summary.csv"
    canonical_json = output_dir / "benchmark_summary.json"
    canonical_csv = output_dir / "benchmark_summary.csv"

    write_json(run_json, summary)
    write_json(canonical_json, summary)
    write_csv(run_csv, rows)
    write_csv(canonical_csv, rows)

    # Canonical plots expected by docs.
    plot_success_rate(rows, output_dir / "success_rate.png")
    plot_time_to_goal(rows, output_dir / "time_to_goal.png")
    plot_safety_hits(rows, output_dir / "safety_hits.png")

    print(f"Wrote {run_json}")
    print(f"Wrote {run_csv}")
    print(f"Wrote {canonical_json}")
    print(f"Wrote {canonical_csv}")


def load_records(input_dir: Path) -> list[EpisodeRecord]:
    records: list[EpisodeRecord] = []
    for path in sorted(input_dir.glob("episodes_*.jsonl")):
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError:
                    continue
                reason = str(raw.get("reason", "")).strip().lower()
                if reason not in ("goal", "stuck"):
                    continue
                scenario = str(raw.get("scenario", "")).strip() or "default"
                planner_mode = str(raw.get("planner_mode", "")).strip() or "unknown"
                record = EpisodeRecord(
                    scenario=scenario,
                    planner_mode=planner_mode,
                    reason=reason,
                    goal_reached=bool(raw.get("goal_reached", reason == "goal")),
                    runtime_sec=float(raw.get("runtime_sec", 0.0)),
                    distance_traveled_m=float(raw.get("distance_traveled_m", 0.0)),
                    goal_distance_m=float(raw.get("goal_distance_m", -1.0)),
                    safety_interventions=int(raw.get("safety_interventions", 0)),
                    collision_flag=bool(raw.get("collision_flag", False)),
                    min_obstacle_range_m=float(raw.get("min_obstacle_range_m", -1.0)),
                )
                records.append(record)
    return records


def aggregate(records: Iterable[EpisodeRecord]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[EpisodeRecord]] = defaultdict(list)
    for rec in records:
        grouped[(rec.scenario, rec.planner_mode)].append(rec)

    rows: list[dict[str, Any]] = []
    for (scenario, planner_mode), items in grouped.items():
        total = len(items)
        success_count = sum(1 for it in items if it.goal_reached or it.reason == "goal")
        success_rate = success_count / total if total else 0.0
        successful_times = [it.runtime_sec for it in items if it.goal_reached or it.reason == "goal"]
        mean_time_to_goal = fmean(successful_times) if successful_times else None
        collision_or_safety_count = sum(
            1 for it in items if it.collision_flag or it.safety_interventions > 0
        )
        collision_or_safety_rate = collision_or_safety_count / total if total else 0.0
        mean_distance = fmean([it.distance_traveled_m for it in items]) if total else 0.0
        mean_safety_hits = fmean([float(it.safety_interventions) for it in items]) if total else 0.0

        rows.append(
            {
                "scenario": scenario,
                "planner_mode": planner_mode,
                "episode_count": total,
                "success_rate": round(success_rate, 6),
                "mean_time_to_goal_sec": None if mean_time_to_goal is None else round(mean_time_to_goal, 4),
                "collision_or_safety_intervention_rate": round(collision_or_safety_rate, 6),
                "mean_distance_traveled_m": round(mean_distance, 4),
                "mean_safety_interventions": round(mean_safety_hits, 4),
            }
        )
    return sorted(rows, key=sort_key)


def sort_key(row: dict[str, Any]) -> tuple[int, int]:
    scenario_order = {"easy": 0, "medium": 1, "cluttered": 2}
    planner_order = {"rule": 0, "rl": 1}
    return (
        scenario_order.get(str(row["scenario"]), 99),
        planner_order.get(str(row["planner_mode"]), 99),
    )


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "scenario",
        "planner_mode",
        "episode_count",
        "success_rate",
        "mean_time_to_goal_sec",
        "collision_or_safety_intervention_rate",
        "mean_distance_traveled_m",
        "mean_safety_interventions",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot_success_rate(rows: list[dict[str, Any]], out_path: Path) -> None:
    plot_grouped_bars(
        rows=rows,
        out_path=out_path,
        metric_key="success_rate",
        title="Success Rate by Scenario",
        ylabel="Success Rate",
    )


def plot_time_to_goal(rows: list[dict[str, Any]], out_path: Path) -> None:
    plot_grouped_bars(
        rows=rows,
        out_path=out_path,
        metric_key="mean_time_to_goal_sec",
        title="Mean Time-to-Goal by Scenario",
        ylabel="Seconds",
        fill_none_with=0.0,
    )


def plot_safety_hits(rows: list[dict[str, Any]], out_path: Path) -> None:
    plot_grouped_bars(
        rows=rows,
        out_path=out_path,
        metric_key="mean_safety_interventions",
        title="Mean Safety Interventions by Scenario",
        ylabel="Interventions / Episode",
    )


def plot_grouped_bars(
    rows: list[dict[str, Any]],
    out_path: Path,
    metric_key: str,
    title: str,
    ylabel: str,
    fill_none_with: float | None = None,
) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
        import numpy as np  # type: ignore
    except ImportError:
        print(f"[WARN] matplotlib/numpy not available, skipping plot {out_path.name}")
        return

    scenarios = sorted({str(r["scenario"]) for r in rows}, key=lambda x: {"easy": 0, "medium": 1, "cluttered": 2}.get(x, 99))
    planners = sorted({str(r["planner_mode"]) for r in rows}, key=lambda x: {"rule": 0, "rl": 1}.get(x, 99))
    if not scenarios or not planners:
        return

    values = {planner: [] for planner in planners}
    for scenario in scenarios:
        for planner in planners:
            match = next(
                (r for r in rows if r["scenario"] == scenario and r["planner_mode"] == planner),
                None,
            )
            value = None if match is None else match.get(metric_key)
            if value is None and fill_none_with is not None:
                value = fill_none_with
            values[planner].append(float(value or 0.0))

    x = np.arange(len(scenarios))
    width = 0.36 if len(planners) == 2 else 0.8 / max(len(planners), 1)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    for idx, planner in enumerate(planners):
        offset = (idx - (len(planners) - 1) / 2.0) * width
        ax.bar(x + offset, values[planner], width=width, label=planner.upper())

    ax.set_title(title)
    ax.set_xlabel("Scenario")
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios)
    ax.legend()
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
