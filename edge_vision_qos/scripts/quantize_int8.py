#!/usr/bin/env python3
"""Create quantized ONNX model (dynamic quantization)."""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input ONNX model")
    parser.add_argument("--output", default="artifacts/yolov8n.int8.onnx")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        from onnxruntime.quantization import QuantType, quantize_dynamic
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("Install onnxruntime-tools support: pip install onnxruntime") from exc

    quantize_dynamic(
        model_input=str(in_path),
        model_output=str(out_path),
        weight_type=QuantType.QInt8,
    )
    print(f"Quantized model written to: {out_path}")


if __name__ == "__main__":
    main()
