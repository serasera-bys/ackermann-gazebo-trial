# Results

Benchmark command (run from `edge_vision_qos/`):

```bash
PYTHONPATH=. .venv/bin/python scripts/benchmark.py \
  --source /home/bernard/video/highway_night.mp4 \
  --model artifacts/yolov8n.onnx \
  --duration-sec 10 --runs-per-scenario 2 \
  --output-csv artifacts/benchmark_summary_baseline.csv

PYTHONPATH=. .venv/bin/python scripts/benchmark.py \
  --source /home/bernard/video/highway_night.mp4 \
  --model artifacts/yolov8n.int8.onnx \
  --duration-sec 10 --runs-per-scenario 2 \
  --output-csv artifacts/benchmark_summary_quantized.csv
```

Aggregated summary (`artifacts/benchmark_summary.csv`):

| Variant | Scenario | Runs | FPS Mean | Latency p95 Mean (ms) | Drop Rate Mean | Deadline Miss Mean | Mean Confidence | Blur Score Mean |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | normal | 2 | 14.69 | 52.56 | 0.0000 | 0.0000 | 0.3632 | 23.8166 |
| baseline | crowded | 2 | 13.32 | 640.74 | 0.0563 | 0.9755 | 0.3636 | 23.7235 |
| baseline | low_light | 2 | 14.98 | 53.64 | 0.0000 | 0.0000 | 0.3477 | 2.3289 |
| quantized_int8 | normal | 2 | 11.53 | 698.83 | 0.1722 | 0.9881 | 0.3707 | 23.5888 |
| quantized_int8 | crowded | 2 | 8.17 | 997.98 | 0.3987 | 0.9945 | 0.3732 | 23.5192 |
| quantized_int8 | low_light | 2 | 10.24 | 794.36 | 0.2616 | 0.9910 | 0.3559 | 2.3175 |

## Notes
- On this setup, dynamic INT8 quantization is slower/worse than baseline ONNX for this workload.
- Runtime metrics are stored per session at `artifacts/metrics_run_<timestamp>.json`.
- Alert/event logs are stored at `artifacts/events_run_<timestamp>.jsonl`.
