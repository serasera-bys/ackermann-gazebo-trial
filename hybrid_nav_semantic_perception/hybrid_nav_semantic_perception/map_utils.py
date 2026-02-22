from __future__ import annotations

import math
from collections import defaultdict
from typing import Iterable

from .projection_utils import map_index


def update_semantic_cells(
    semantic_data: list[int],
    width: int,
    height: int,
    resolution: float,
    origin_x: float,
    origin_y: float,
    objects: Iterable[dict],
) -> tuple[list[int], dict[int, dict[str, float | str]]]:
    updated = list(semantic_data)
    object_cells: dict[int, dict[str, float | str]] = {}
    for obj in objects:
        x = float(obj.get("map_x", 0.0))
        y = float(obj.get("map_y", 0.0))
        score = float(obj.get("score", 0.0))
        class_id = str(obj.get("class_id", "unknown"))

        idx = map_index(x, y, resolution, width, height, origin_x, origin_y)
        if idx is None:
            continue

        score_100 = int(max(0.0, min(1.0, score if math.isfinite(score) else 0.0)) * 100.0)
        updated[idx] = max(updated[idx], score_100)
        object_cells[idx] = {
            "class_id": class_id,
            "score": float(score),
            "map_x": x,
            "map_y": y,
            "map_z": float(obj.get("map_z", 0.0)),
        }

    return updated, object_cells


def class_counts(cells: dict[int, dict[str, float | str]]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for entry in cells.values():
        counts[str(entry.get("class_id", "unknown"))] += 1
    return dict(counts)
