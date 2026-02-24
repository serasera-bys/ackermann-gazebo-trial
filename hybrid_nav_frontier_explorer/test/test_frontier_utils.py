import numpy as np

from hybrid_nav_frontier_explorer.frontier_utils import (
    cluster_centroid_world,
    cluster_frontiers,
    compute_reachable_free_mask,
    compute_frontier_mask,
    find_nearest_free_cell,
    has_occupied_within,
    is_point_blacklisted,
    prune_expired_points,
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


def test_reachable_free_mask_respects_walls() -> None:
    grid = np.zeros((7, 7), dtype=np.int16)
    grid[:, 3] = 100
    mask = compute_reachable_free_mask(grid, 1, 1)
    assert mask[1, 1] == 1
    assert mask[1, 5] == 0


def test_find_nearest_free_cell() -> None:
    grid = np.full((5, 5), 100, dtype=np.int16)
    grid[2, 3] = 0
    found = find_nearest_free_cell(grid, 2, 2, max_radius=2)
    assert found == (3, 2)


def test_has_occupied_within_clearance() -> None:
    grid = np.zeros((9, 9), dtype=np.int16)
    grid[4, 6] = 100
    assert has_occupied_within(grid, 4, 4, clearance_cells=3)
    assert not has_occupied_within(grid, 1, 1, clearance_cells=1)


def test_blacklist_prune_and_query() -> None:
    points = [
        {"x": 1.0, "y": 2.0, "expire": 5.0},
        {"x": -1.0, "y": -1.0, "expire": 1.0},
    ]
    alive = prune_expired_points(points, now_sec=2.0)
    assert len(alive) == 1
    assert is_point_blacklisted(alive, x=1.1, y=2.0, radius=0.25)
    assert not is_point_blacklisted(alive, x=2.0, y=2.0, radius=0.25)
