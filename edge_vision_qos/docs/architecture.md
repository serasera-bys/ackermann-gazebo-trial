# Edge Vision QoS Monitor Architecture

## Dataflow
`source -> capture loop -> bounded queue -> detector loop -> QoS monitor -> artifacts/API`

## Embedded-style Decisions
- Bounded queue (`max_queue_size`) to cap memory and bound latency.
- Drop-oldest policy on overflow to keep freshest frames.
- Target-rate capture loop (`target_fps`) to enforce timing budget.
- Deadline miss tracking (`deadline_ms`) as watchdog-style signal.
- Alert thresholds for latency/FPS/drop/blur deterioration.

## Components
- `app/pipeline.py`: session lifecycle, capture/inference threads.
- `app/detector.py`: ONNX YOLO detector with mock fallback.
- `app/qos_monitor.py`: runtime metrics and alert logic.
- `app/api.py`: FastAPI endpoints.
- `scripts/benchmark.py`: reproducible scenario runs.
