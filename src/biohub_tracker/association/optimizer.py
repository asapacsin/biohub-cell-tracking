from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp  # type: ignore[import-untyped]
from scipy.sparse import lil_matrix  # type: ignore[import-untyped]

from biohub_tracker.association.candidates import SparseCandidateGraph


@dataclass(frozen=True, slots=True)
class OptimizerConfig:
    method: str = "ilp"
    minimum_score: float = 0.0
    time_limit_seconds: float = 120.0

    def __post_init__(self) -> None:
        if self.method not in {"ilp", "greedy"}:
            raise ValueError("optimizer method must be 'ilp' or 'greedy'")
        if self.time_limit_seconds <= 0:
            raise ValueError("time_limit_seconds must be positive")


def _event_edges(graph: SparseCandidateGraph) -> list[tuple[tuple[int, int], ...]]:
    events: list[tuple[tuple[int, int], ...]] = [
        ((edge.source, edge.target),) for edge in graph.edges
    ]
    events.extend(
        ((division.source, division.child_a), (division.source, division.child_b))
        for division in graph.divisions
    )
    return events


def _greedy_selection(
    graph: SparseCandidateGraph, config: OptimizerConfig
) -> list[tuple[int, int]]:
    events = _event_edges(graph)
    scores = [edge.score for edge in graph.edges] + [item.score for item in graph.divisions]
    used_sources: set[int] = set()
    used_targets: set[int] = set()
    selected: list[tuple[int, int]] = []
    for index in sorted(range(len(events)), key=lambda value: (-scores[value], value)):
        event = events[index]
        if scores[index] < config.minimum_score:
            continue
        source = event[0][0]
        targets = {target for _, target in event}
        if source in used_sources or targets & used_targets:
            continue
        used_sources.add(source)
        used_targets.update(targets)
        selected.extend(event)
    return selected


def optimize_candidate_graph(
    graph: SparseCandidateGraph, config: OptimizerConfig
) -> list[tuple[int, int]]:
    """Select globally consistent continuation/gap/division events.

    A node receives at most one parent and emits at most one event. A division event atomically
    emits exactly two edges, so lineage branching cannot be half-selected.
    """
    events = _event_edges(graph)
    if not events:
        return []
    if config.method == "greedy":
        return _greedy_selection(graph, config)

    scores = np.asarray(
        [edge.score for edge in graph.edges] + [item.score for item in graph.divisions],
        dtype=np.float64,
    )
    event_count = len(events)
    rows: list[tuple[str, int]] = []
    rows.extend(("source", index) for index in sorted({event[0][0] for event in events}))
    target_indices = {pair[1] for event in events for pair in event}
    rows.extend(("target", index) for index in sorted(target_indices))
    matrix = lil_matrix((len(rows), event_count), dtype=np.float64)
    for row, (kind, node_index) in enumerate(rows):
        for column, event in enumerate(events):
            is_source = kind == "source" and event[0][0] == node_index
            is_target = kind == "target" and any(target == node_index for _, target in event)
            if is_source or is_target:
                matrix[row, column] = 1.0
    constraints = LinearConstraint(matrix.tocsr(), 0.0, 1.0)
    result = milp(
        c=-scores,
        integrality=np.ones(event_count),
        bounds=Bounds(np.zeros(event_count), np.ones(event_count)),
        constraints=constraints,
        options={"time_limit": config.time_limit_seconds},
    )
    if not result.success or result.x is None:
        return _greedy_selection(graph, config)
    selected: list[tuple[int, int]] = []
    for index, value in enumerate(result.x):
        if value >= 0.5 and scores[index] >= config.minimum_score:
            selected.extend(events[index])
    return selected
