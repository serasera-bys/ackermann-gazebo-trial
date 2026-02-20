from app.qos_monitor import AlertThresholds, QosMonitor


def test_alerts_trigger_for_latency_and_drop_rate() -> None:
    monitor = QosMonitor(
        target_fps=15.0,
        deadline_ms=120.0,
        thresholds=AlertThresholds(latency_p95_ms=20.0, drop_rate_max=0.1, blur_consecutive_frames=3),
    )

    for _ in range(10):
        monitor.record_ingest(queue_depth=2)
    for _ in range(4):
        monitor.record_drop(queue_depth=8)

    alerts = []
    for _ in range(5):
        alerts = monitor.record_processed(
            latency_ms=50.0,
            queue_delay_ms=10.0,
            queue_depth=4,
            mean_confidence=0.7,
            blur_score=20.0,
            deadline_missed=True,
        )

    alert_types = {a["type"] for a in alerts}
    assert "latency_p95_high" in alert_types
    assert "drop_rate_high" in alert_types
    assert "blur_persistent" in alert_types


def test_live_metrics_shape() -> None:
    monitor = QosMonitor(target_fps=15.0, deadline_ms=120.0)
    monitor.record_ingest(queue_depth=1)
    monitor.record_processed(10.0, 2.0, 1, 0.8, 80.0, False)

    live = monitor.live_metrics()
    assert "fps_actual" in live
    assert "latency_ms_p95" in live
    assert "frame_drop_rate" in live
    assert "deadline_miss_rate" in live
