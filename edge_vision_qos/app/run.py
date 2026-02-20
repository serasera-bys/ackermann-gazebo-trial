from __future__ import annotations

import argparse
import time

from .pipeline import SessionConfig, VisionPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Edge Vision QoS CLI runner")
    parser.add_argument("--source", required=True, help="Video path/RTSP URL/camera index")
    parser.add_argument("--target-fps", type=float, default=15.0)
    parser.add_argument("--max-queue-size", type=int, default=8)
    parser.add_argument("--deadline-ms", type=float, default=120.0)
    parser.add_argument("--model-path", default="")
    parser.add_argument("--conf-threshold", type=float, default=0.25)
    parser.add_argument("--artifact-dir", default="artifacts")
    parser.add_argument("--duration-sec", type=float, default=30.0)
    parser.add_argument("--simulated-inference-delay-ms", type=float, default=0.0)
    parser.add_argument("--force-blur", action="store_true")
    parser.add_argument("--print-every-sec", type=float, default=2.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = SessionConfig(
        source=args.source,
        target_fps=args.target_fps,
        max_queue_size=args.max_queue_size,
        deadline_ms=args.deadline_ms,
        model_path=args.model_path,
        conf_threshold=args.conf_threshold,
        artifact_dir=args.artifact_dir,
        simulated_inference_delay_ms=args.simulated_inference_delay_ms,
        force_blur=args.force_blur,
    )

    pipeline = VisionPipeline()
    info = pipeline.start(cfg)
    print(f"Session started: {info}")

    started = time.monotonic()
    next_print = started
    try:
        while (time.monotonic() - started) < max(0.1, args.duration_sec):
            if not pipeline.is_running():
                break
            now = time.monotonic()
            if now >= next_print:
                print(pipeline.live_metrics())
                next_print = now + max(0.2, args.print_every_sec)
            time.sleep(0.05)
    finally:
        stop_info = pipeline.stop()
        print(f"Session stopped: {stop_info}")
        print("Final metrics:", pipeline.live_metrics())


if __name__ == "__main__":
    main()
