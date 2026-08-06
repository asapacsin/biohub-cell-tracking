from __future__ import annotations

from biohub_pipeline import postprocessing
from biohub_pipeline.config import load_config


def _stats() -> dict[str, int]:
    return {
        "short_track_filter_skipped_all": 0,
        "short_track_rescue_triggered": 0,
        "short_track_rescue_budget": 0,
        "short_track_rescue_components": 0,
        "short_track_rescue_nodes": 0,
        "short_track_components_removed": 0,
        "short_track_nodes_removed": 0,
        "short_track_edges_removed": 0,
    }


def test_high_confidence_five_node_rescue() -> None:
    config = load_config("configs/clean_v106.yaml")
    settings = dict(config.postprocessing)
    settings["short_track_rescue_max_nodes_frac"] = 1.0
    postprocessing.configure(settings)
    nodes = {
        **{i: {"node_id": i, "t": i, "z": 1.0, "y": float(i), "x": 1.0} for i in range(6)},
        **{
            100 + i: {"node_id": 100 + i, "t": i, "z": 2.0, "y": float(i), "x": 2.0}
            for i in range(5)
        },
    }
    edges = [
        *[
            {"source_id": i, "target_id": i + 1, "edge_prob": 0.95, "distance_um": 0.5}
            for i in range(5)
        ],
        *[
            {"source_id": 100 + i, "target_id": 101 + i, "edge_prob": 0.95, "distance_um": 0.5}
            for i in range(4)
        ],
    ]
    kept_nodes, kept_edges = postprocessing.filter_short_track_components(nodes, edges, _stats())
    assert len(kept_nodes) == 11
    assert len(kept_edges) == 9


def test_low_confidence_short_track_is_removed() -> None:
    config = load_config("configs/clean_v106.yaml")
    settings = dict(config.postprocessing)
    settings["short_track_rescue_max_nodes_frac"] = 1.0
    postprocessing.configure(settings)
    nodes = {
        **{i: {"node_id": i, "t": i, "z": 1.0, "y": float(i), "x": 1.0} for i in range(6)},
        **{
            100 + i: {"node_id": 100 + i, "t": i, "z": 2.0, "y": float(i), "x": 2.0}
            for i in range(5)
        },
    }
    edges = [
        *[
            {"source_id": i, "target_id": i + 1, "edge_prob": 0.95, "distance_um": 0.5}
            for i in range(5)
        ],
        *[
            {"source_id": 100 + i, "target_id": 101 + i, "edge_prob": 0.2, "distance_um": 0.5}
            for i in range(4)
        ],
    ]
    kept_nodes, _ = postprocessing.filter_short_track_components(nodes, edges, _stats())
    assert set(kept_nodes) == set(range(6))
