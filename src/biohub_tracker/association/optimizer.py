from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp  # type: ignore[import-untyped]
from scipy.sparse import coo_matrix  # type: ignore[import-untyped]

from biohub_tracker.association.candidates import SparseCandidateGraph

# SciPy HiGHS struggles past this event count on CPU; keep architecture runnable.
_DEFAULT_ILP_EVENT_LIMIT = 40_000


@dataclass(frozen=True, slots=True)
class OptimizerConfig:
    method: str = "ilp"
    minimum_score: float = 0.0
    time_limit_seconds: float = 120.0
    ilp_event_limit: int = _DEFAULT_ILP_EVENT_LIMIT

    def __post_init__(self) -> None:
        if self.method not in {"ilp", "greedy"}:
            raise ValueError("optimizer method must be 'ilp' or 'greedy'")
        if self.time_limit_seconds <= 0:
            raise ValueError("time_limit_seconds must be positive")
        if self.ilp_event_limit < 1:
            raise ValueError("ilp_event_limit must be positive")


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


def _constraint_matrix(events: list[tuple[tuple[int, int], ...]]) -> coo_matrix:
    """Build sparse <=1 parent/child incidence without an O(rows*events) Python loop."""
    source_nodes = sorted({event[0][0] for event in events})
    target_nodes = sorted({target for event in events for _, target in event})
    source_row = {node: index for index, node in enumerate(source_nodes)}
    target_row = {
        node: index + len(source_nodes) for index, node in enumerate(target_nodes)
    }
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    for column, event in enumerate(events):
        rows.append(source_row[event[0][0]])
        cols.append(column)
        data.append(1.0)
        for _, target in event:
            rows.append(target_row[target])
            cols.append(column)
            data.append(1.0)
    return coo_matrix(
        (data, (rows, cols)),
        shape=(len(source_nodes) + len(target_nodes), len(events)),
        dtype=np.float64,
    )


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
    if config.method == "greedy" or len(events) > config.ilp_event_limit:
        return _greedy_selection(graph, config)

    scores = np.asarray(
        [edge.score for edge in graph.edges] + [item.score for item in graph.divisions],
        dtype=np.float64,
    )
    constraints = LinearConstraint(_constraint_matrix(events).tocsr(), 0.0, 1.0)
    result = milp(
        c=-scores,
        integrality=np.ones(len(events)),
        bounds=Bounds(np.zeros(len(events)), np.ones(len(events))),
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
