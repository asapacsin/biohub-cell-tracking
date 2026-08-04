from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from biohub_tracker.association.candidates import SparseCandidateGraph


@dataclass(frozen=True, slots=True)
class CandidateLabels:
    edge_labels: NDArray[np.int8]
    division_labels: NDArray[np.int8]


def label_candidate_graph(
    graph: SparseCandidateGraph,
    *,
    detection_to_geff_node: dict[int, int],
    geff_edges: set[tuple[int, int]],
) -> CandidateLabels:
    """Label sparse candidates against matched GEFF nodes without inventing missing labels."""
    edge_labels = np.full(len(graph.edges), -1, dtype=np.int8)
    for index, edge in enumerate(graph.edges):
        source = detection_to_geff_node.get(edge.source)
        target = detection_to_geff_node.get(edge.target)
        if source is not None and target is not None:
            edge_labels[index] = int((source, target) in geff_edges)

    outgoing: dict[int, set[int]] = {}
    for source, target in geff_edges:
        outgoing.setdefault(source, set()).add(target)
    division_labels = np.full(len(graph.divisions), -1, dtype=np.int8)
    for index, division in enumerate(graph.divisions):
        source = detection_to_geff_node.get(division.source)
        child_a = detection_to_geff_node.get(division.child_a)
        child_b = detection_to_geff_node.get(division.child_b)
        if source is not None and child_a is not None and child_b is not None:
            children = outgoing.get(source, set())
            division_labels[index] = int({child_a, child_b} <= children)
    return CandidateLabels(edge_labels=edge_labels, division_labels=division_labels)
