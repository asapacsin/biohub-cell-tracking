from __future__ import annotations

import csv

import pytest

from biohub_pipeline.submission import (
    CSV_COLUMNS,
    graph_rows,
    validate_graph,
    validate_submission_file,
    write_rows,
)


def _division_graph():
    nodes = {
        10: {"node_id": 10, "t": 0, "z": 1.2, "y": 2.4, "x": 3.6},
        11: {"node_id": 11, "t": 1, "z": 1.0, "y": 2.0, "x": 3.0},
        12: {"node_id": 12, "t": 1, "z": 2.0, "y": 3.0, "x": 4.0},
    }
    edges = [{"source_id": 10, "target_id": 11}, {"source_id": 10, "target_id": 12}]
    return nodes, edges


def test_coordinate_order_video_frames_track_ids_and_division(tmp_path) -> None:
    nodes, edges = _division_graph()
    rows = graph_rows("video_a", nodes, edges)
    assert [row["id"] for row in rows] == list(range(5))
    assert rows[0]["dataset"] == "video_a"
    assert (rows[0]["z"], rows[0]["y"], rows[0]["x"]) == (1, 2, 4)
    assert {row["node_id"] for row in rows[:3]} == {10, 11, 12}
    assert {(row["source_id"], row["target_id"]) for row in rows[3:]} == {(10, 11), (10, 12)}
    path = write_rows(rows, tmp_path / "submission.csv")
    with path.open(newline="", encoding="utf-8") as handle:
        assert next(csv.reader(handle)) == CSV_COLUMNS
    assert validate_submission_file(path) == {"rows": 5, "datasets": 1, "nodes": 3, "edges": 2}


def test_multiple_parents_are_rejected() -> None:
    nodes, _ = _division_graph()
    nodes[13] = {"node_id": 13, "t": 0, "z": 1, "y": 1, "x": 1}
    with pytest.raises(ValueError, match="more than one parent"):
        validate_graph(
            "video", nodes, [{"source_id": 10, "target_id": 11}, {"source_id": 13, "target_id": 11}]
        )


def test_slightly_negative_floats_are_clamped_like_upstream() -> None:
    nodes = {
        1: {"node_id": 1, "t": 0, "z": -0.4, "y": 1.0, "x": 2.0},
        2: {"node_id": 2, "t": 1, "z": 0.2, "y": 1.0, "x": 2.0},
    }
    edges = [{"source_id": 1, "target_id": 2}]
    rows = graph_rows("video_b", nodes, edges)
    assert rows[0]["z"] == 0
    assert rows[1]["z"] == 0
