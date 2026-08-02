from __future__ import annotations

from biohub_tracker.models import PredictedEdge, PredictedNode, PredictionGraph


def tiny_expected_graph(dataset: str = "tiny") -> PredictionGraph:
    """Deterministic graph fixture; it is not produced by a detector or tracker."""
    nodes = [
        PredictedNode(dataset, 1, 0, 2, 10, 10),
        PredictedNode(dataset, 2, 0, 3, 20, 20),
        PredictedNode(dataset, 3, 1, 2, 11, 10),
        PredictedNode(dataset, 4, 1, 3, 21, 20),
        PredictedNode(dataset, 5, 2, 2, 12, 9),
        PredictedNode(dataset, 6, 2, 2, 12, 12),
        PredictedNode(dataset, 7, 2, 3, 22, 20),
    ]
    edges = [
        PredictedEdge(dataset, 1, 3),
        PredictedEdge(dataset, 2, 4),
        PredictedEdge(dataset, 3, 5),
        PredictedEdge(dataset, 3, 6),
        PredictedEdge(dataset, 4, 7),
    ]
    return PredictionGraph(nodes=nodes, edges=edges)

