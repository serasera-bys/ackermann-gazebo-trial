from __future__ import annotations

from typing import Iterable

import numpy as np

from .schema import SensorSample

FEATURE_NAMES = [
    "s1_mean",
    "s2_mean",
    "s3_mean",
    "s4_mean",
    "s5_mean",
    "s1_std",
    "s2_std",
    "s3_std",
    "s4_std",
    "s5_std",
    "s1_slope",
    "s2_slope",
    "s3_slope",
    "s4_slope",
    "s5_slope",
    "delta_s1",
    "delta_s2",
    "delta_s3",
    "delta_s4",
    "delta_s5",
]


def samples_to_matrix(samples: Iterable[SensorSample]) -> np.ndarray:
    rows = [[s.timestamp_sec, s.s1, s.s2, s.s3, s.s4, s.s5] for s in samples]
    arr = np.asarray(rows, dtype=float)
    if arr.shape[0] < 5:
        raise ValueError("Need at least 5 samples per window")
    return arr


def build_features(samples: Iterable[SensorSample]) -> np.ndarray:
    arr = samples_to_matrix(samples)
    t = arr[:, 0]
    sensors = arr[:, 1:]

    means = np.mean(sensors, axis=0)
    stds = np.std(sensors, axis=0)

    slopes = []
    t0 = t - t.min()
    denom = float(np.sum((t0 - t0.mean()) ** 2))
    if denom <= 1e-9:
        denom = 1.0
    for i in range(sensors.shape[1]):
        y = sensors[:, i]
        slope = float(np.sum((t0 - t0.mean()) * (y - y.mean())) / denom)
        slopes.append(slope)

    deltas = sensors[-1, :] - sensors[0, :]

    vec = np.concatenate([means, stds, np.asarray(slopes), deltas])
    return vec.astype(float)


def bucketize_rul_risk(risk_score: float) -> str:
    if risk_score >= 0.70:
        return "high"
    if risk_score >= 0.35:
        return "medium"
    return "low"
