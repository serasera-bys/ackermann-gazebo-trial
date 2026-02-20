from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .pipeline import SessionConfig, VisionPipeline

app = FastAPI(title="Edge Vision QoS Monitor", version="0.1.0")
PIPELINE = VisionPipeline()


class SessionStartRequest(BaseModel):
    source: str = Field(..., description="Video file path, RTSP URL, or camera index string")
    target_fps: float = 15.0
    max_queue_size: int = 8
    deadline_ms: float = 120.0
    model_path: str = ""
    conf_threshold: float = 0.25
    artifact_dir: str = "artifacts"
    simulated_inference_delay_ms: float = 0.0
    force_blur: bool = False


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "session_running": PIPELINE.is_running()}


@app.post("/session/start")
def session_start(req: SessionStartRequest) -> dict:
    try:
        cfg = SessionConfig(**req.model_dump())
        return PIPELINE.start(cfg)
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.post("/session/stop")
def session_stop() -> dict:
    return PIPELINE.stop()


@app.get("/metrics/live")
def metrics_live() -> dict:
    return PIPELINE.live_metrics()


@app.get("/metrics/history")
def metrics_history() -> dict:
    return PIPELINE.metrics_history()
