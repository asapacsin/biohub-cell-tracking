from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass

from biohub_tracker.models import PredictedEdge, PredictionGraph


@dataclass(frozen=True, slots=True)
class PostprocessingConfig:
    minimum_track_length: int = 1
    remove_isolated_short_tracks: bool = False

    def __post_init__(self) -> None:
        if self.minimum_track_length < 1:
            raise ValueError("minimum_track_length must be >= 1")


def _weak_components(graph: PredictionGraph) -> list[set[int]]:
    adjacency: dict[int, set[int]] = defaultdict(set)
    for edge in graph.edges:
        adjacency[edge.source_id].add(edge.target_id)
        adjacency[edge.target_id].add(edge.source_id)
    unseen = {node.node_id for node in graph.nodes}
    components: list[set[int]] = []
    while unseen:
        start = min(unseen)
        component: set[int] = set()
        queue = deque([start])
        while queue:
            node_id = queue.popleft()
            if node_id in component:
                continue
            component.add(node_id)
            queue.extend(adjacency[node_id] - component)
        unseen -= component
        components.append(component)
    return components


def postprocess_prediction_graph(
    graph: PredictionGraph, config: PostprocessingConfig
) -> PredictionGraph:
    """Remove duplicate edges and, when requested, isolated short track components."""
    unique_edges = sorted(
        {(edge.source_id, edge.target_id) for edge in graph.edges}, key=lambda item: item
    )
    cleaned = PredictionGraph(
        nodes=list(graph.nodes),
        edges=[
            PredictedEdge(dataset=graph.nodes[0].dataset, source_id=source, target_id=target)
            for source, target in unique_edges
        ]
        if graph.nodes
        else [],
    )
    if not config.remove_isolated_short_tracks or config.minimum_track_length <= 1:
        return cleaned
    keep = {
        node_id
        for component in _weak_components(cleaned)
        if len(component) >= config.minimum_track_length
        for node_id in component
    }
    return PredictionGraph(
        nodes=[node for node in cleaned.nodes if node.node_id in keep],
        edges=[edge for edge in cleaned.edges if edge.source_id in keep and edge.target_id in keep],
    )
