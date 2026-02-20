from __future__ import annotations

import statistics
import threading
import time
from collections import deque
from dataclasses import dataclass


@dataclass(slots=True)
class AlertThresholds:
    latency_p95_ms: float = 120.0
    fps_ratio_floor: float = 0.7
    drop_rate_max: float = 0.15
    blur_score_min: float = 40.0
    blur_consecutive_frames: int = 10


class QosMonitor:
    def __init__(
        self,
        target_fps: float,
        deadline_ms: float,
        thresholds: AlertThresholds | None = None,
        history_limit: int = 360,
    ) -> None:
        self.target_fps = max(1e-6, target_fps)
        self.deadline_ms = deadline_ms
        self.thresholds = thresholds or AlertThresholds()
        self.history_limit = history_limit

        self._lock = threading.Lock()
        self._start = time.monotonic()

        self.frames_ingested = 0
        self.frames_processed = 0
        self.frames_dropped = 0
        self.deadline_misses = 0

        self.latencies = deque(maxlen=4000)
        self.queue_delays = deque(maxlen=4000)
        self.queue_depths = deque(maxlen=4000)
        self.confidences = deque(maxlen=4000)
        self.blur_scores = deque(maxlen=4000)
        self.alert_history = deque(maxlen=history_limit)

        self._blur_low_streak = 0

    def record_ingest(self, queue_depth: int) -> None:
        with self._lock:
            self.frames_ingested += 1
            self.queue_depths.append(float(queue_depth))

    def record_drop(self, queue_depth: int) -> None:
        with self._lock:
            self.frames_dropped += 1
            self.queue_depths.append(float(queue_depth))

    def record_processed(
        self,
        latency_ms: float,
        queue_delay_ms: float,
        queue_depth: int,
        mean_confidence: float,
        blur_score: float,
        deadline_missed: bool,
    ) -> list[dict]:
        with self._lock:
            self.frames_processed += 1
            self.latencies.append(float(latency_ms))
            self.queue_delays.append(float(queue_delay_ms))
            self.queue_depths.append(float(queue_depth))
            self.confidences.append(float(mean_confidence))
            self.blur_scores.append(float(blur_score))
            if deadline_missed:
                self.deadline_misses += 1

            if blur_score < self.thresholds.blur_score_min:
                self._blur_low_streak += 1
            else:
                self._blur_low_streak = 0

            alerts = self._evaluate_alerts_locked()
            for alert in alerts:
                payload = {
                    "ts": time.time(),
                    "type": alert["type"],
                    "value": alert["value"],
                    "threshold": alert["threshold"],
                }
                self.alert_history.append(payload)
            return alerts

    def live_metrics(self) -> dict:
        with self._lock:
            elapsed = max(1e-6, time.monotonic() - self._start)
            fps_actual = self.frames_processed / elapsed
            latency_p50 = percentile(self.latencies, 50.0)
            latency_p95 = percentile(self.latencies, 95.0)
            queue_avg = average(self.queue_depths)
            queue_max = max(self.queue_depths) if self.queue_depths else 0.0
            frame_drop_rate = self.frames_dropped / max(1, self.frames_ingested)
            deadline_miss_rate = self.deadline_misses / max(1, self.frames_processed)
            mean_conf = average(self.confidences)
            blur_score = average(self.blur_scores)

            return {
                "uptime_sec": elapsed,
                "target_fps": self.target_fps,
                "fps_actual": fps_actual,
                "latency_ms_p50": latency_p50,
                "latency_ms_p95": latency_p95,
                "frame_drop_rate": frame_drop_rate,
                "queue_depth_avg": queue_avg,
                "queue_depth_max": queue_max,
                "mean_confidence": mean_conf,
                "blur_score": blur_score,
                "deadline_miss_rate": deadline_miss_rate,
                "frames_ingested": self.frames_ingested,
                "frames_processed": self.frames_processed,
                "frames_dropped": self.frames_dropped,
                "deadline_misses": self.deadline_misses,
                "active_alerts": self._evaluate_alerts_locked(),
            }

    def history(self) -> dict:
        return {
            "alerts": list(self.alert_history),
            "live": self.live_metrics(),
        }

    def _evaluate_alerts_locked(self) -> list[dict]:
        elapsed = max(1e-6, time.monotonic() - self._start)
        fps_actual = self.frames_processed / elapsed
        latency_p95 = percentile(self.latencies, 95.0)
        frame_drop_rate = self.frames_dropped / max(1, self.frames_ingested)
        alerts: list[dict] = []

        if latency_p95 > self.thresholds.latency_p95_ms:
            alerts.append(
                {
                    "type": "latency_p95_high",
                    "value": latency_p95,
                    "threshold": self.thresholds.latency_p95_ms,
                }
            )
        if fps_actual < (self.target_fps * self.thresholds.fps_ratio_floor):
            alerts.append(
                {
                    "type": "fps_low",
                    "value": fps_actual,
                    "threshold": self.target_fps * self.thresholds.fps_ratio_floor,
                }
            )
        if frame_drop_rate > self.thresholds.drop_rate_max:
            alerts.append(
                {
                    "type": "drop_rate_high",
                    "value": frame_drop_rate,
                    "threshold": self.thresholds.drop_rate_max,
                }
            )
        if self._blur_low_streak >= self.thresholds.blur_consecutive_frames:
            alerts.append(
                {
                    "type": "blur_persistent",
                    "value": self._blur_low_streak,
                    "threshold": self.thresholds.blur_consecutive_frames,
                }
            )
        return alerts


def average(values: deque[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def percentile(values: deque[float], q: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    # statistics.quantiles is stable and no numpy dependency here.
    n = max(2, min(100, int(round(q))))
    try:
        bins = statistics.quantiles(values, n=n, method="inclusive")
        idx = max(0, min(len(bins) - 1, int(round(q)) - 1))
        return float(bins[idx])
    except statistics.StatisticsError:
        ordered = sorted(values)
        pos = int((q / 100.0) * (len(ordered) - 1))
        return float(ordered[pos])
