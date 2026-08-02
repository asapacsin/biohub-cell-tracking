from biohub_tracker.models import PredictedEdge, PredictedNode, PredictionGraph
from biohub_tracker.submission.validator import validate_graph


def test_one_continuation_is_one_edge_with_new_target_id() -> None:
    graph = PredictionGraph(
        nodes=[PredictedNode("a", 10, 0, 1, 2, 3), PredictedNode("a", 25, 1, 1, 2, 4)],
        edges=[PredictedEdge("a", 10, 25)],
    )
    validate_graph(graph)
    assert graph.edges == [PredictedEdge("a", 10, 25)]

