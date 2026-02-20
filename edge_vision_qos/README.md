# Edge Vision QoS Monitor

Realtime object-detection pipeline with embedded-style reliability monitoring.

## Features
- ONNX runtime inference (YOLOv8 ONNX) with mock fallback.
- Bounded queue + frame drop policy.
- QoS metrics: FPS, latency p50/p95, drop rate, queue depth, confidence, blur score, deadline miss rate.
- Alerting rules for latency/FPS/drop/blur degradation.
- REST API + CLI runner.

## Quickstart
```bash
cd edge_vision_qos
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run CLI session:
```bash
python3 -m app.run --source path/to/video.mp4 --target-fps 15 --duration-sec 30
```

`app.run` prints QoS metrics/events to terminal and writes artifacts. It does not open a live video window.
To generate an annotated detection video:
```bash
yolo predict task=detect model=artifacts/yolov8n.onnx source=path/to/video.mp4 save=True
```
Ultralytics output is saved under `runs/detect/predict*/`.

Run API:
```bash
uvicorn app.api:app --reload --port 8080
```

## API
- `POST /session/start`
- `POST /session/stop`
- `GET /metrics/live`
- `GET /metrics/history`
- `GET /health`

## Scripts
Export YOLOv8 to ONNX:
```bash
python3 scripts/export_onnx.py --model yolov8n.pt --output artifacts/yolov8n.onnx
```

Quantize ONNX:
```bash
python3 scripts/quantize_int8.py --input artifacts/yolov8n.onnx --output artifacts/yolov8n.int8.onnx
```

Benchmark scenarios:
```bash
PYTHONPATH=. python3 scripts/benchmark.py --source path/to/video.mp4 --model artifacts/yolov8n.onnx
```

Run tests:
```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q
```

## Artifacts
- `artifacts/metrics_run_<timestamp>.json`
- `artifacts/events_run_<timestamp>.jsonl`
- `artifacts/benchmark_summary.csv`

See docs:
- `docs/architecture.md`
- `docs/benchmark_protocol.md`
- `docs/results.md`
