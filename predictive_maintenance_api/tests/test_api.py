from app.main import health, model_info, predict
from app.schema import PredictRequest


def test_health_and_model_info() -> None:
    health_payload = health()
    assert health_payload["status"] == "ok"

    info_payload = model_info().model_dump()
    assert "feature_count" in info_payload


def test_predict_contract() -> None:
    payload = PredictRequest(
        unit_id="u1",
        window=[
            {"timestamp_sec": 0, "s1": 0.1, "s2": 0.2, "s3": 0.3, "s4": 0.4, "s5": 0.5},
            {"timestamp_sec": 1, "s1": 0.2, "s2": 0.2, "s3": 0.3, "s4": 0.4, "s5": 0.5},
            {"timestamp_sec": 2, "s1": 0.3, "s2": 0.2, "s3": 0.3, "s4": 0.4, "s5": 0.5},
            {"timestamp_sec": 3, "s1": 0.4, "s2": 0.2, "s3": 0.3, "s4": 0.4, "s5": 0.5},
            {"timestamp_sec": 4, "s1": 0.5, "s2": 0.2, "s3": 0.3, "s4": 0.4, "s5": 0.5}
        ]
    )

    body = predict(payload).model_dump()
    assert 0.0 <= body["risk_score"] <= 1.0
    assert body["remaining_useful_life_bucket"] in {"low", "medium", "high"}
