from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class SensorSample(BaseModel):
    timestamp_sec: float = Field(..., ge=0.0)
    s1: float
    s2: float
    s3: float
    s4: float
    s5: float


class PredictRequest(BaseModel):
    unit_id: str = "unit"
    window: list[SensorSample] = Field(..., min_length=5)


class PredictResponse(BaseModel):
    unit_id: str
    risk_score: float
    remaining_useful_life_bucket: Literal["low", "medium", "high"]


class BatchPredictRequest(BaseModel):
    items: list[PredictRequest] = Field(..., min_length=1)


class ModelInfoResponse(BaseModel):
    model_path: str
    feature_count: int
    trained_rows: int
    algorithm: str
