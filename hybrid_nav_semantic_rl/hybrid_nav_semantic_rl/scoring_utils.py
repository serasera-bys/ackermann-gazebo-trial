from __future__ import annotations

import json
import math
from typing import Iterable

FEATURE_ORDER = [
    "distance_to_frontier",
    "estimated_free_gain",
    "heading_change",
    "local_obstacle_risk",
    "semantic_novelty_score",
    "semantic_priority_score",
]


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def yaw_from_quaternion(x: float, y: float, z: float, w: float) -> float:
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def heading_delta(cur_yaw: float, target_x: float, target_y: float, cur_x: float, cur_y: float) -> float:
    target_yaw = math.atan2(target_y - cur_y, target_x - cur_x)
    d = target_yaw - cur_yaw
    while d > math.pi:
        d -= 2.0 * math.pi
    while d < -math.pi:
        d += 2.0 * math.pi
    return abs(d)


def parse_class_weights(raw: str) -> dict[str, float]:
    text = raw.strip()
    if not text:
        return {}
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return {}
    if not isinstance(data, dict):
        return {}
    out: dict[str, float] = {}
    for k, v in data.items():
        try:
            out[str(k)] = float(v)
        except (TypeError, ValueError):
            continue
    return out


def novelty_from_semantic_grid(values: Iterable[int]) -> float:
    valid = [v for v in values if v >= 0]
    if not valid:
        return 1.0
    mean_conf = sum(valid) / len(valid)
    return clamp(1.0 - (mean_conf / 100.0), 0.0, 1.0)


def priority_from_objects(
    candidate_x: float,
    candidate_y: float,
    objects: list[dict],
    class_weights: dict[str, float],
    radius: float = 1.5,
) -> float:
    if not objects:
        return 0.0
    total = 0.0
    count = 0
    for obj in objects:
        ox = float(obj.get("map_x", 0.0))
        oy = float(obj.get("map_y", 0.0))
        dist = math.hypot(ox - candidate_x, oy - candidate_y)
        if dist > radius:
            continue
        class_id = str(obj.get("class_id", "unknown"))
        weight = class_weights.get(class_id, 0.0)
        conf = clamp(float(obj.get("score", 0.0)), 0.0, 1.0)
        total += weight * conf
        count += 1
    if count == 0:
        return 0.0
    return clamp(total / count, 0.0, 1.0)


def rule_score(
    features: dict[str, float],
    w1: float,
    w2: float,
    w3: float,
    w4: float,
    w5: float,
) -> float:
    return (
        w1 * features["estimated_free_gain"]
        + w2 * features["semantic_novelty_score"]
        + w3 * features["semantic_priority_score"]
        - w4 * features["distance_to_frontier"]
        - w5 * features["local_obstacle_risk"]
    )


def linear_policy_score(features: dict[str, float], weights: dict[str, float], bias: float) -> float:
    return bias + sum(float(weights.get(k, 0.0)) * float(features[k]) for k in FEATURE_ORDER)
