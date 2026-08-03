from __future__ import annotations

from biohub_tracker.models import PredictedNode
from biohub_tracker.tracking import TrackingConfig, link_consecutive_nodes


def test_linker_connects_nearby_nodes() -> None:
    previous = [
        PredictedNode("tiny", 1, 0, 2, 10, 10),
        PredictedNode("tiny", 2, 0, 3, 20, 20),
    ]
    current = [
        PredictedNode("tiny", 3, 1, 2, 11, 10),
        PredictedNode("tiny", 4, 1, 3, 21, 20),
    ]
    edges = link_consecutive_nodes(
        previous,
        current,
        voxel_spacing_zyx=(1.625, 0.40625, 0.40625),
        config=TrackingConfig(max_match_distance_um=10.0, candidate_neighbors=3),
    )
    assert {(edge.source_id, edge.target_id) for edge in edges} == {(1, 3), (2, 4)}


def test_linker_rejects_far_nodes() -> None:
    previous = [PredictedNode("tiny", 1, 0, 0, 0, 0)]
    current = [PredictedNode("tiny", 2, 1, 50, 200, 200)]
    edges = link_consecutive_nodes(
        previous,
        current,
        voxel_spacing_zyx=(1.625, 0.40625, 0.40625),
        config=TrackingConfig(max_match_distance_um=5.0),
    )
    assert edges == []
