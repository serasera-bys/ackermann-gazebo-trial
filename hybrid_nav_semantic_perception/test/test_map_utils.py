from hybrid_nav_semantic_perception.map_utils import update_semantic_cells


def test_update_semantic_cells_marks_grid() -> None:
    semantic = [-1] * 100
    updated, cells = update_semantic_cells(
        semantic_data=semantic,
        width=10,
        height=10,
        resolution=1.0,
        origin_x=0.0,
        origin_y=0.0,
        objects=[{"class_id": "chair", "score": 0.9, "map_x": 2.1, "map_y": 3.2, "map_z": 0.1}],
    )
    idx = 3 * 10 + 2
    assert updated[idx] == 90
    assert idx in cells
    assert cells[idx]["class_id"] == "chair"
