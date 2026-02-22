from __future__ import annotations

from collections import deque

import numpy as np


UNKNOWN = -1
FREE_MAX = 10
OCCUPIED_MIN = 50


def occupancy_to_grid(data: list[int], width: int, height: int) -> np.ndarray:
    return np.asarray(data, dtype=np.int16).reshape((height, width))


def compute_frontier_mask(grid: np.ndarray) -> np.ndarray:
    h, w = grid.shape
    frontier = np.zeros((h, w), dtype=np.uint8)
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            if grid[y, x] < 0 or grid[y, x] > FREE_MAX:
                continue
            neigh = grid[y - 1 : y + 2, x - 1 : x + 2]
            if np.any(neigh == UNKNOWN):
                frontier[y, x] = 1
    return frontier


def cluster_frontiers(mask: np.ndarray, min_cluster_size: int = 5) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    visited = np.zeros_like(mask, dtype=np.uint8)
    clusters: list[list[tuple[int, int]]] = []

    for y in range(h):
        for x in range(w):
            if mask[y, x] == 0 or visited[y, x] == 1:
                continue

            q = deque([(x, y)])
            visited[y, x] = 1
            cluster: list[tuple[int, int]] = []

            while q:
                cx, cy = q.popleft()
                cluster.append((cx, cy))
                for ny in range(max(0, cy - 1), min(h, cy + 2)):
                    for nx in range(max(0, cx - 1), min(w, cx + 2)):
                        if visited[ny, nx] == 1 or mask[ny, nx] == 0:
                            continue
                        visited[ny, nx] = 1
                        q.append((nx, ny))

            if len(cluster) >= min_cluster_size:
                clusters.append(cluster)

    return clusters


def cluster_centroid_world(
    cluster: list[tuple[int, int]],
    resolution: float,
    origin_x: float,
    origin_y: float,
) -> tuple[float, float]:
    sx = sum(p[0] for p in cluster)
    sy = sum(p[1] for p in cluster)
    n = max(1, len(cluster))
    gx = sx / n
    gy = sy / n
    wx = origin_x + (gx + 0.5) * resolution
    wy = origin_y + (gy + 0.5) * resolution
    return wx, wy
