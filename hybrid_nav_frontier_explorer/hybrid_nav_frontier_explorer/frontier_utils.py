from __future__ import annotations

from collections import deque

import numpy as np


UNKNOWN = -1
FREE_MAX = 10
OCCUPIED_MIN = 50


def occupancy_to_grid(data: list[int], width: int, height: int) -> np.ndarray:
    return np.asarray(data, dtype=np.int16).reshape((height, width))


def world_to_grid(
    wx: float,
    wy: float,
    resolution: float,
    origin_x: float,
    origin_y: float,
    width: int,
    height: int,
) -> tuple[int, int] | None:
    gx = int((wx - origin_x) / max(resolution, 1e-9))
    gy = int((wy - origin_y) / max(resolution, 1e-9))
    if gx < 0 or gy < 0 or gx >= width or gy >= height:
        return None
    return gx, gy


def _is_free_cell(grid: np.ndarray, x: int, y: int) -> bool:
    val = int(grid[y, x])
    return 0 <= val <= FREE_MAX


def find_nearest_free_cell(
    grid: np.ndarray,
    start_x: int,
    start_y: int,
    max_radius: int = 8,
) -> tuple[int, int] | None:
    h, w = grid.shape
    if start_x < 0 or start_y < 0 or start_x >= w or start_y >= h:
        return None
    if _is_free_cell(grid, start_x, start_y):
        return start_x, start_y

    for radius in range(1, max_radius + 1):
        min_x = max(0, start_x - radius)
        max_x = min(w - 1, start_x + radius)
        min_y = max(0, start_y - radius)
        max_y = min(h - 1, start_y + radius)
        for y in range(min_y, max_y + 1):
            for x in range(min_x, max_x + 1):
                if _is_free_cell(grid, x, y):
                    return x, y
    return None


def compute_reachable_free_mask(grid: np.ndarray, start_x: int, start_y: int) -> np.ndarray:
    h, w = grid.shape
    reachable = np.zeros((h, w), dtype=np.uint8)
    if start_x < 0 or start_y < 0 or start_x >= w or start_y >= h:
        return reachable
    if not _is_free_cell(grid, start_x, start_y):
        return reachable

    q = deque([(start_x, start_y)])
    reachable[start_y, start_x] = 1
    while q:
        cx, cy = q.popleft()
        for ny in range(max(0, cy - 1), min(h, cy + 2)):
            for nx in range(max(0, cx - 1), min(w, cx + 2)):
                if reachable[ny, nx] == 1:
                    continue
                if not _is_free_cell(grid, nx, ny):
                    continue
                reachable[ny, nx] = 1
                q.append((nx, ny))
    return reachable


def has_occupied_within(grid: np.ndarray, x: int, y: int, clearance_cells: int) -> bool:
    if clearance_cells <= 0:
        return False
    h, w = grid.shape
    min_x = max(0, x - clearance_cells)
    max_x = min(w - 1, x + clearance_cells)
    min_y = max(0, y - clearance_cells)
    max_y = min(h - 1, y + clearance_cells)
    max_dist2 = clearance_cells * clearance_cells
    for yy in range(min_y, max_y + 1):
        dy = yy - y
        for xx in range(min_x, max_x + 1):
            dx = xx - x
            if (dx * dx + dy * dy) > max_dist2:
                continue
            if int(grid[yy, xx]) >= OCCUPIED_MIN:
                return True
    return False


def prune_expired_points(points: list[dict[str, float]], now_sec: float) -> list[dict[str, float]]:
    return [p for p in points if float(p.get("expire", -1.0)) > now_sec]


def is_point_blacklisted(
    points: list[dict[str, float]],
    x: float,
    y: float,
    radius: float,
) -> bool:
    if radius <= 0.0:
        return False
    for item in points:
        bx = float(item.get("x", float("nan")))
        by = float(item.get("y", float("nan")))
        if not np.isfinite(bx) or not np.isfinite(by):
            continue
        if np.hypot(x - bx, y - by) <= radius:
            return True
    return False


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
