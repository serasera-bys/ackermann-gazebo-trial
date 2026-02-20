#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

from app.pipeline import SessionConfig, VisionPipeline

SCENARIOS = {
    "normal": {"simulated_inference_delay_ms": 0.0, "force_blur": False},
    "crowded": {"simulated_inference_delay_ms": 35.0, "force_blur": False},
    "low_light": {"simulated_inference_delay_ms": 10.0, "force_blur": True},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark edge vision QoS")
    parser.add_argument("--source", required=True)
    parser.add_argument("--model", default="")
    parser.add_argument("--duration-sec", type=float, default=20.0)
    parser.add_argument("--runs-per-scenario", type=int, default=3)
    parser.add_argument("--target-fps", type=float, default=15.0)
    parser.add_argument("--artifact-dir", default="artifacts")
    parser.add_argument("--output-csv", default="artifacts/benchmark_summary.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_csv = Path(args.output_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for scenario, opts in SCENARIOS.items():
        for run_idx in range(1, args.runs_per_scenario + 1):
            pipeline = VisionPipeline()
            cfg = SessionConfig(
                source=args.source,
                model_path=args.model,
                target_fps=args.target_fps,
                artifact_dir=args.artifact_dir,
                simulated_inference_delay_ms=opts["simulated_inference_delay_ms"],
                force_blur=opts["force_blur"],
            )
            pipeline.start(cfg)
            started = time.monotonic()
            while (time.monotonic() - started) < args.duration_sec and pipeline.is_running():
                time.sleep(0.1)
            pipeline.stop()

            metrics = pipeline.live_metrics()
            row = {
                "scenario": scenario,
                "run": run_idx,
                "fps_actual": metrics.get("fps_actual", 0.0),
                "latency_ms_p95": metrics.get("latency_ms_p95", 0.0),
                "frame_drop_rate": metrics.get("frame_drop_rate", 0.0),
                "deadline_miss_rate": metrics.get("deadline_miss_rate", 0.0),
                "mean_confidence": metrics.get("mean_confidence", 0.0),
                "blur_score": metrics.get("blur_score", 0.0),
            }
            rows.append(row)
            print(json.dumps(row))

    headers = [
        "scenario",
        "run",
        "fps_actual",
        "latency_ms_p95",
        "frame_drop_rate",
        "deadline_miss_rate",
        "mean_confidence",
        "blur_score",
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote benchmark summary: {out_csv}")


if __name__ == "__main__":
    main()
