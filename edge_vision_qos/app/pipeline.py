from __future__ import annotations

import json
import queue
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .detector import Detection, build_detector
from .qos_monitor import AlertThresholds, QosMonitor


@dataclass(slots=True)
class SessionConfig:
    source: str
    target_fps: float = 15.0
    max_queue_size: int = 8
    deadline_ms: float = 120.0
    model_path: str = ""
    conf_threshold: float = 0.25
    artifact_dir: str = "artifacts"
    simulated_inference_delay_ms: float = 0.0
    force_blur: bool = False


class VisionPipeline:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._running = False
        self._capture_thread: threading.Thread | None = None
        self._infer_thread: threading.Thread | None = None
        self._frame_queue: queue.Queue = queue.Queue(maxsize=8)
        self._monitor: QosMonitor | None = None
        self._events: list[dict] = []
        self._session_id = ""
        self._config: SessionConfig | None = None
        self._detector = None

    def start(self, cfg: SessionConfig) -> dict:
        with self._lock:
            if self._running:
                raise RuntimeError("session already running")
            self._running = True
            self._config = cfg
            self._session_id = time.strftime("%Y%m%d_%H%M%S")
            self._events = []
            self._frame_queue = queue.Queue(maxsize=max(1, cfg.max_queue_size))
            self._monitor = QosMonitor(
                target_fps=cfg.target_fps,
                deadline_ms=cfg.deadline_ms,
                thresholds=AlertThresholds(),
            )
            self._detector = build_detector(cfg.model_path, cfg.conf_threshold)

        self._capture_thread = threading.Thread(target=self._capture_loop, name="capture", daemon=True)
        self._infer_thread = threading.Thread(target=self._infer_loop, name="infer", daemon=True)
        self._capture_thread.start()
        self._infer_thread.start()
        return {"session_id": self._session_id, "status": "running"}

    def stop(self) -> dict:
        with self._lock:
            was_running = self._running
            self._running = False
        if not was_running:
            return {"status": "idle"}

        if self._capture_thread is not None:
            self._capture_thread.join(timeout=2.0)
        if self._infer_thread is not None:
            self._infer_thread.join(timeout=2.0)

        self._write_artifacts()
        return {"status": "stopped", "session_id": self._session_id}

    def is_running(self) -> bool:
        with self._lock:
            return self._running

    def live_metrics(self) -> dict:
        monitor = self._monitor
        if monitor is None:
            return {
                "status": "idle",
                "active_alerts": [],
            }
        metrics = monitor.live_metrics()
        metrics["status"] = "running" if self.is_running() else "stopped"
        metrics["session_id"] = self._session_id
        return metrics

    def metrics_history(self) -> dict:
        monitor = self._monitor
        if monitor is None:
            return {"status": "idle", "alerts": [], "live": {}}
        data = monitor.history()
        data["status"] = "running" if self.is_running() else "stopped"
        data["session_id"] = self._session_id
        return data

    def _capture_loop(self) -> None:
        cfg = self._config
        if cfg is None or self._monitor is None:
            return
        try:
            import cv2
        except ImportError:
            self._events.append(
                {
                    "ts": time.time(),
                    "type": "runtime_error",
                    "message": "OpenCV not installed; capture loop cannot run.",
                }
            )
            with self._lock:
                self._running = False
            return

        cap = cv2.VideoCapture(cfg.source)
        if not cap.isOpened():
            self._events.append(
                {
                    "ts": time.time(),
                    "type": "runtime_error",
                    "message": f"Failed to open source: {cfg.source}",
                }
            )
            with self._lock:
                self._running = False
            return

        target_period = 1.0 / max(1e-6, cfg.target_fps)
        next_tick = time.monotonic()

        while self.is_running():
            now = time.monotonic()
            if now < next_tick:
                time.sleep(min(0.002, next_tick - now))
                continue
            next_tick += target_period

            ok, frame = cap.read()
            if not ok:
                break
            frame_ts = time.monotonic()
            try:
                self._frame_queue.put_nowait((frame_ts, frame))
            except queue.Full:
                _ = self._frame_queue.get_nowait()
                self._frame_queue.put_nowait((frame_ts, frame))
                self._monitor.record_drop(self._frame_queue.qsize())
            self._monitor.record_ingest(self._frame_queue.qsize())

        cap.release()
        with self._lock:
            self._running = False

    def _infer_loop(self) -> None:
        cfg = self._config
        monitor = self._monitor
        if cfg is None or monitor is None:
            return

        while self.is_running() or (not self._frame_queue.empty()):
            try:
                frame_ts, frame = self._frame_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if cfg.force_blur:
                frame = self._apply_blur(frame)

            start = time.monotonic()
            queue_delay_ms = (start - frame_ts) * 1000.0
            detections: list[Detection] = self._detector.infer(frame)
            if cfg.simulated_inference_delay_ms > 0.0:
                time.sleep(cfg.simulated_inference_delay_ms / 1000.0)
            end = time.monotonic()
            latency_ms = (end - frame_ts) * 1000.0
            deadline_missed = latency_ms > cfg.deadline_ms

            blur_score = self._blur_score(frame)
            mean_conf = float(np.mean([d.confidence for d in detections])) if detections else 0.0

            alerts = monitor.record_processed(
                latency_ms=latency_ms,
                queue_delay_ms=queue_delay_ms,
                queue_depth=self._frame_queue.qsize(),
                mean_confidence=mean_conf,
                blur_score=blur_score,
                deadline_missed=deadline_missed,
            )
            for alert in alerts:
                self._events.append({"ts": time.time(), **alert})

    @staticmethod
    def _apply_blur(frame: np.ndarray) -> np.ndarray:
        try:
            import cv2

            return cv2.GaussianBlur(frame, (11, 11), 0)
        except ImportError:
            return frame

    @staticmethod
    def _blur_score(frame: np.ndarray) -> float:
        try:
            import cv2

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            return float(cv2.Laplacian(gray, cv2.CV_64F).var())
        except ImportError:
            return 0.0

    def _write_artifacts(self) -> None:
        cfg = self._config
        monitor = self._monitor
        if cfg is None or monitor is None:
            return

        out_dir = Path(cfg.artifact_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        metrics_path = out_dir / f"metrics_run_{self._session_id}.json"
        events_path = out_dir / f"events_run_{self._session_id}.jsonl"

        metrics = {
            "session_id": self._session_id,
            "config": asdict(cfg),
            "metrics": monitor.live_metrics(),
            "history": monitor.history(),
            "event_count": len(self._events),
        }
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

        with events_path.open("w", encoding="utf-8") as f:
            for event in self._events:
                f.write(json.dumps(event) + "\n")

        self._events.append(
            {
                "ts": time.time(),
                "type": "artifacts_written",
                "metrics_path": str(metrics_path),
                "events_path": str(events_path),
            }
        )
