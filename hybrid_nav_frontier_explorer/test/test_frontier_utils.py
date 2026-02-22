import numpy as np

from hybrid_nav_frontier_explorer.frontier_utils import (
    cluster_centroid_world,
    cluster_frontiers,
    compute_frontier_mask,
)


def test_frontier_extraction_simple_band() -> None:
    grid = np.full((10, 10), -1, dtype=np.int16)
    grid[5:, :] = 0
    mask = compute_frontier_mask(grid)
    clusters = cluster_frontiers(mask, min_cluster_size=3)
    assert clusters


def test_cluster_centroid_world() -> None:
    cluster = [(2, 4), (3, 4), (4, 4)]
    x, y = cluster_centroid_world(cluster, resolution=0.5, origin_x=1.0, origin_y=-1.0)
    assert 2.0 <= x <= 3.5
    assert 1.0 <= y <= 1.5
