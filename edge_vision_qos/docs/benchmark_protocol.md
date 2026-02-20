# Benchmark Protocol

## Scenarios
- `normal`: baseline runtime.
- `crowded`: injected higher inference delay.
- `low_light`: injected blur + mild delay.

## Command
```bash
cd edge_vision_qos
python3 scripts/benchmark.py \
  --source path/to/video.mp4 \
  --model artifacts/yolov8n.onnx \
  --runs-per-scenario 3 \
  --duration-sec 20
```

## Key Metrics
- `fps_actual`
- `latency_ms_p95`
- `frame_drop_rate`
- `queue_depth_avg/max` (from per-run metrics json)
- `mean_confidence`
- `blur_score`
- `deadline_miss_rate`

## Outputs
- `artifacts/benchmark_summary.csv`
- `artifacts/metrics_run_<timestamp>.json`
- `artifacts/events_run_<timestamp>.jsonl`
