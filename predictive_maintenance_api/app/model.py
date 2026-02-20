from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np

from .features import FEATURE_NAMES


@dataclass(slots=True)
class ModelBundle:
    model: object
    algorithm: str
    trained_rows: int
    model_path: str


class HeuristicFallbackModel:
    """Fallback used if trained model artifact is missing."""

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        # Simple monotonic heuristic from sensor deltas + std.
        scores = np.clip((np.mean(np.abs(x[:, -5:]), axis=1) + np.mean(x[:, 5:10], axis=1)) / 10.0, 0.0, 1.0)
        return np.vstack([1.0 - scores, scores]).T


def load_model_bundle(model_path: str, metadata_path: str) -> ModelBundle:
    mp = Path(model_path)
    md = Path(metadata_path)

    if not mp.exists() or not md.exists():
        return ModelBundle(
            model=HeuristicFallbackModel(),
            algorithm="heuristic_fallback",
            trained_rows=0,
            model_path=str(mp),
        )

    model = joblib.load(mp)
    metadata = json.loads(md.read_text(encoding="utf-8"))
    return ModelBundle(
        model=model,
        algorithm=str(metadata.get("algorithm", "unknown")),
        trained_rows=int(metadata.get("trained_rows", 0)),
        model_path=str(mp),
    )


def predict_risk(model: object, feature_vec: np.ndarray) -> float:
    x = feature_vec.reshape(1, -1)
    if hasattr(model, "predict_proba"):
        y = model.predict_proba(x)
        return float(np.clip(y[0, 1], 0.0, 1.0))
    # fallback if estimator only supports predict
    y = model.predict(x)
    return float(np.clip(float(y[0]), 0.0, 1.0))


def write_feature_schema(path: str) -> None:
    payload = {
        "feature_names": FEATURE_NAMES,
        "feature_count": len(FEATURE_NAMES),
    }
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
