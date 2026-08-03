from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment  # type: ignore[import-untyped]

from biohub_tracker.coordinates import physical_distance_zyx
from biohub_tracker.models import PredictedEdge, PredictedNode


@dataclass(frozen=True, slots=True)
class TrackingConfig:
    candidate_neighbors: int = 5
    max_match_distance_um: float = 15.0
    use_hungarian: bool = True
    distance_weight: float = 1.0
    intensity_weight: float = 0.0
    volume_weight: float = 0.0

    def __post_init__(self) -> None:
        if self.candidate_neighbors < 1:
            raise ValueError("candidate_neighbors must be >= 1")
        if self.max_match_distance_um <= 0:
            raise ValueError("max_match_distance_um must be positive")
        if self.distance_weight < 0 or self.intensity_weight < 0 or self.volume_weight < 0:
            raise ValueError("tracking weights must be non-negative")


def _cost(
    source: PredictedNode,
    target: PredictedNode,
    *,
    voxel_spacing_zyx: tuple[float, float, float],
    config: TrackingConfig,
) -> float:
    distance = physical_distance_zyx(
        (source.z, source.y, source.x),
        (target.z, target.y, target.x),
        voxel_spacing_zyx,
    )
    return float(config.distance_weight * distance)


def link_consecutive_nodes(
    previous: list[PredictedNode],
    current: list[PredictedNode],
    *,
    voxel_spacing_zyx: tuple[float, float, float],
    config: TrackingConfig,
) -> list[PredictedEdge]:
    if not previous or not current:
        return []

    n_prev = len(previous)
    n_curr = len(current)
    large = config.max_match_distance_um * 1000.0 + 1.0
    cost = np.full((n_prev, n_curr), large, dtype=np.float64)
    for i, source in enumerate(previous):
        distances = [
            (
                j,
                _cost(
                    source,
                    target,
                    voxel_spacing_zyx=voxel_spacing_zyx,
                    config=config,
                ),
            )
            for j, target in enumerate(current)
        ]
        distances.sort(key=lambda item: item[1])
        for j, value in distances[: config.candidate_neighbors]:
            if value <= config.max_match_distance_um:
                cost[i, j] = value

    edges: list[PredictedEdge] = []
    if config.use_hungarian:
        row_ind, col_ind = linear_sum_assignment(cost)
        pairs = list(zip(row_ind, col_ind, strict=True))
    else:
        pairs = []
        used_targets: set[int] = set()
        order = sorted(
            ((i, j, cost[i, j]) for i in range(n_prev) for j in range(n_curr)),
            key=lambda item: item[2],
        )
        used_sources: set[int] = set()
        for i, j, value in order:
            if value > config.max_match_distance_um:
                break
            if i in used_sources or j in used_targets:
                continue
            pairs.append((i, j))
            used_sources.add(i)
            used_targets.add(j)

    dataset = previous[0].dataset
    for i, j in pairs:
        if cost[i, j] > config.max_match_distance_um:
            continue
        edges.append(
            PredictedEdge(
                dataset=dataset,
                source_id=previous[i].node_id,
                target_id=current[j].node_id,
            )
        )
    return edges
