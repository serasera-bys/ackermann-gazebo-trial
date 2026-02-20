#!/usr/bin/env python3
"""Export YOLOv8n to ONNX for CPU inference."""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--output", default="artifacts/yolov8n.onnx")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        from ultralytics import YOLO
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("Install ultralytics first: pip install ultralytics") from exc

    model = YOLO(args.model)
    export_path = model.export(format="onnx", imgsz=args.imgsz, simplify=True)
    export_path = Path(export_path)
    export_path.replace(out_path)
    print(f"Exported ONNX model to: {out_path}")


if __name__ == "__main__":
    main()
