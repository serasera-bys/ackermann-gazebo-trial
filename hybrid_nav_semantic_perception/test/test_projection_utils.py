from hybrid_nav_semantic_perception.projection_utils import depth_to_camera_point, transform_point


def test_depth_to_camera_point_center() -> None:
    x, y, z = depth_to_camera_point(320.0, 240.0, 2.0, 500.0, 500.0, 320.0, 240.0)
    assert abs(x) < 1e-6
    assert abs(y) < 1e-6
    assert abs(z - 2.0) < 1e-6


def test_transform_point_translation_only() -> None:
    x, y, z = transform_point((1.0, 2.0, 3.0), (0.5, -1.0, 2.0), (0.0, 0.0, 0.0, 1.0))
    assert abs(x - 1.5) < 1e-6
    assert abs(y - 1.0) < 1e-6
    assert abs(z - 5.0) < 1e-6
