from __future__ import annotations

from biohub_tracker.baseline_pipeline import observations_to_graph
from biohub_tracker.tracking.data_types import TrackObservation


def _obs(
    frame: int,
    det_id: int,
    cell_id: int,
    z: float,
    y: float,
    x: float,
    *,
    parent_id: int | None = None,
    link_status: str = "initial",
) -> TrackObservation:
    return TrackObservation(
        "v",
        frame,
        det_id,
        cell_id,
        z,
        y,
        x,
        parent_id=parent_id,
        link_status=link_status,
        division_score=1.5 if link_status == "division_child" else None,
    )


def test_continuation_edge() -> None:
    observations = [
        _obs(0, 0, 1, 0, 0, 0, link_status="initial"),
        _obs(1, 0, 1, 0, 1, 0, link_status="matched"),
    ]
    graph, stamped = observations_to_graph(observations, "dataset")
    assert len(graph.nodes) == 2
    assert len(graph.edges) == 1
    assert graph.edges[0].source_id == stamped[0].node_id
    assert graph.edges[0].target_id == stamped[1].node_id


def test_division_emits_two_edges_from_parent_node() -> None:
    observations = [
        _obs(0, 0, 5, 0, 0, 0, link_status="initial"),
        _obs(1, 0, 10, 0, -1, 0, parent_id=5, link_status="division_child"),
        _obs(1, 1, 11, 0, 1, 0, parent_id=5, link_status="division_child"),
    ]
    graph, stamped = observations_to_graph(observations, "dataset")
    parent_node = next(n.node_id for n in stamped if n.cell_id == 5)
    daughter_nodes = sorted(n.node_id for n in stamped if n.link_status == "division_child")
    edge_pairs = sorted((e.source_id, e.target_id) for e in graph.edges)
    assert edge_pairs == [
        (parent_node, daughter_nodes[0]),
        (parent_node, daughter_nodes[1]),
    ]
    assert all(n.division_score == 1.5 for n in stamped if n.link_status == "division_child")
