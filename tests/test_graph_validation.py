import pytest

from biohub_tracker.fixtures import tiny_expected_graph
from biohub_tracker.models import PredictedEdge, PredictedNode, PredictionGraph
from biohub_tracker.submission.validator import ValidationError, validate_graph


def test_valid_tiny_graph_and_division(tiny_metadata) -> None:
    graph = tiny_expected_graph()
    validate_graph(graph, {"tiny": tiny_metadata})
    assert sum(edge.source_id == 3 for edge in graph.edges) == 2


def test_source_and_target_must_exist() -> None:
    graph = PredictionGraph(
        nodes=[PredictedNode("a", 1, 0, 0, 0, 0)],
        edges=[PredictedEdge("a", 1, 99)],
    )
    with pytest.raises(ValidationError, match="missing target"):
        validate_graph(graph)


def test_duplicate_edges_are_rejected() -> None:
    graph = PredictionGraph(
        nodes=[PredictedNode("a", 1, 0, 0, 0, 0), PredictedNode("a", 2, 1, 0, 0, 0)],
        edges=[PredictedEdge("a", 1, 2), PredictedEdge("a", 1, 2)],
    )
    with pytest.raises(ValidationError, match="duplicate edge"):
        validate_graph(graph)


def test_cycles_are_rejected() -> None:
    graph = PredictionGraph(
        nodes=[PredictedNode("a", 1, 0, 0, 0, 0), PredictedNode("a", 2, 1, 0, 0, 0)],
        edges=[PredictedEdge("a", 1, 2), PredictedEdge("a", 2, 1)],
    )
    with pytest.raises(ValidationError):
        validate_graph(graph)


def test_invalid_coordinate_is_rejected(tiny_metadata) -> None:
    graph = PredictionGraph(nodes=[PredictedNode("tiny", 1, 0, 4, 0, 0)])
    with pytest.raises(ValidationError, match="outside"):
        validate_graph(graph, {"tiny": tiny_metadata})


def test_skipped_empty_frame_is_supported() -> None:
    graph = PredictionGraph(
        nodes=[PredictedNode("a", 1, 0, 0, 0, 0), PredictedNode("a", 2, 2, 0, 0, 0)],
        edges=[PredictedEdge("a", 1, 2)],
    )
    validate_graph(graph)
