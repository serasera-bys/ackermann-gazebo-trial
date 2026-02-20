from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI

from .features import FEATURE_NAMES, bucketize_rul_risk, build_features
from .model import load_model_bundle, predict_risk
from .schema import BatchPredictRequest, ModelInfoResponse, PredictRequest, PredictResponse

ART_DIR = Path(os.environ.get("PM_ARTIFACT_DIR", "artifacts"))
MODEL_PATH = ART_DIR / "model.joblib"
META_PATH = ART_DIR / "train_metadata.json"

app = FastAPI(title="Predictive Maintenance API", version="0.1.0")
BUNDLE = load_model_bundle(str(MODEL_PATH), str(META_PATH))


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "algorithm": BUNDLE.algorithm,
        "model_path": BUNDLE.model_path,
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    features = build_features(req.window)
    risk = predict_risk(BUNDLE.model, features)
    return PredictResponse(
        unit_id=req.unit_id,
        risk_score=risk,
        remaining_useful_life_bucket=bucketize_rul_risk(risk),
    )


@app.post("/batch_predict")
def batch_predict(req: BatchPredictRequest) -> dict:
    items = [predict(item).model_dump() for item in req.items]
    return {"count": len(items), "items": items}


@app.get("/model/info", response_model=ModelInfoResponse)
def model_info() -> ModelInfoResponse:
    return ModelInfoResponse(
        model_path=BUNDLE.model_path,
        feature_count=len(FEATURE_NAMES),
        trained_rows=BUNDLE.trained_rows,
        algorithm=BUNDLE.algorithm,
    )
