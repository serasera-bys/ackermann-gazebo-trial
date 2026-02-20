from app.api import health, metrics_history, metrics_live


def test_health_and_metrics_endpoints() -> None:
    health_payload = health()
    assert "status" in health_payload

    live_payload = metrics_live()
    assert "status" in live_payload

    history_payload = metrics_history()
    assert "alerts" in history_payload
