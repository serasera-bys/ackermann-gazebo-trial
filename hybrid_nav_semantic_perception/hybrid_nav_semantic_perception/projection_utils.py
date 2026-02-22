from __future__ import annotations

import math
from typing import Iterable


def depth_to_camera_point(
    u: float,
    v: float,
    depth_m: float,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> tuple[float, float, float]:
    if depth_m <= 0.0 or not math.isfinite(depth_m):
        raise ValueError("Depth must be finite and > 0")
    x = (u - cx) * depth_m / max(fx, 1e-9)
    y = (v - cy) * depth_m / max(fy, 1e-9)
    z = depth_m
    return x, y, z


def rotate_vector_quaternion(
    vec_xyz: tuple[float, float, float],
    quat_xyzw: tuple[float, float, float, float],
) -> tuple[float, float, float]:
    x, y, z = vec_xyz
    qx, qy, qz, qw = quat_xyzw

    tx = 2.0 * (qy * z - qz * y)
    ty = 2.0 * (qz * x - qx * z)
    tz = 2.0 * (qx * y - qy * x)

    rx = x + qw * tx + (qy * tz - qz * ty)
    ry = y + qw * ty + (qz * tx - qx * tz)
    rz = z + qw * tz + (qx * ty - qy * tx)
    return rx, ry, rz


def transform_point(
    point_xyz: tuple[float, float, float],
    translation_xyz: tuple[float, float, float],
    quat_xyzw: tuple[float, float, float, float],
) -> tuple[float, float, float]:
    rx, ry, rz = rotate_vector_quaternion(point_xyz, quat_xyzw)
    tx, ty, tz = translation_xyz
    return rx + tx, ry + ty, rz + tz


def clamp_pixel(u: float, v: float, width: int, height: int) -> tuple[int, int]:
    uu = int(min(max(round(u), 0), max(width - 1, 0)))
    vv = int(min(max(round(v), 0), max(height - 1, 0)))
    return uu, vv


def sample_depth_window(depth_image, u: int, v: int, window: int = 2) -> float:
    values: list[float] = []
    h, w = depth_image.shape[:2]
    for yy in range(max(0, v - window), min(h, v + window + 1)):
        for xx in range(max(0, u - window), min(w, u + window + 1)):
            val = float(depth_image[yy, xx])
            if math.isfinite(val) and val > 0.05:
                values.append(val)
    if not values:
        raise ValueError("No valid depth in window")
    values.sort()
    return values[len(values) // 2]


def map_index(
    map_x: float,
    map_y: float,
    resolution: float,
    width: int,
    height: int,
    origin_x: float,
    origin_y: float,
) -> int | None:
    gx = int((map_x - origin_x) / max(resolution, 1e-9))
    gy = int((map_y - origin_y) / max(resolution, 1e-9))
    if gx < 0 or gy < 0 or gx >= width or gy >= height:
        return None
    return gy * width + gx


def average(values: Iterable[float]) -> float:
    vals = [float(v) for v in values]
    if not vals:
        return 0.0
    return sum(vals) / len(vals)
