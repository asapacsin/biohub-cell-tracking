from collections import Counter

from biohub_tracker.fixtures import tiny_expected_graph
from biohub_tracker.submission.validator import validate_graph


def test_division_is_two_outgoing_edges() -> None:
    graph = tiny_expected_graph()
    validate_graph(graph)
    outgoing = Counter(edge.source_id for edge in graph.edges)
    assert outgoing[3] == 2
    assert {(edge.source_id, edge.target_id) for edge in graph.edges if edge.source_id == 3} == {
        (3, 5),
        (3, 6),
    }

