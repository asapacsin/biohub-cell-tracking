from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations

import numpy as np

from biohub_tracker.coordinates import physical_distance_zyx
from biohub_tracker.models import DetectionCandidate


@dataclass(frozen=True, slots=True)
class CandidateGraphConfig:
    max_neighbors: int = 8
    max_gap: int = 2
    max_speed_um_per_frame: float = 15.0
    divisions_enabled: bool = True
    max_division_children: int = 6
    max_daughter_separation_um: float = 16.0

    def __post_init__(self) -> None:
        if self.max_neighbors < 1 or self.max_gap < 1:
            raise ValueError("max_neighbors and max_gap must be >= 1")
        if self.max_speed_um_per_frame <= 0 or self.max_daughter_separation_um <= 0:
            raise ValueError("candidate graph distances must be positive")
        if self.max_division_children < 2:
            raise ValueError("max_division_children must be >= 2")


@dataclass(slots=True)
class CandidateEdge:
    source: int
    target: int
    delta_t: int
    distance_um: float
    confidence_mean: float
    displacement_zyx_um: tuple[float, float, float]
    appearance_similarity: float | None = None
    intensity_log_ratio: float | None = None
    volume_log_ratio: float | None = None
    temporal_density_log_ratio: float = 0.0
    score: float = 0.0


@dataclass(slots=True)
class DivisionCandidate:
    source: int
    child_a: int
    child_b: int
    parent_child_distance_mean_um: float
    daughter_separation_um: float
    midpoint_distance_um: float
    confidence_mean: float
    score: float = 0.0


@dataclass(slots=True)
class SparseCandidateGraph:
    nodes: list[DetectionCandidate]
    edges: list[CandidateEdge] = field(default_factory=list)
    divisions: list[DivisionCandidate] = field(default_factory=list)


def _optional_log_ratio(first: float | None, second: float | None) -> float | None:
    if first is None or second is None or first <= 0 or second <= 0:
        return None
    return float(abs(np.log(second / first)))


def _appearance_similarity(
    first: tuple[float, ...] | None, second: tuple[float, ...] | None
) -> float | None:
    if first is None or second is None or len(first) != len(second) or not first:
        return None
    first_array = np.asarray(first, dtype=np.float64)
    second_array = np.asarray(second, dtype=np.float64)
    denominator = float(np.linalg.norm(first_array) * np.linalg.norm(second_array))
    if denominator <= 0:
        return None
    return float(np.dot(first_array, second_array) / denominator)


def build_candidate_graph(
    detections: list[DetectionCandidate],
    *,
    voxel_spacing_zyx: tuple[float, float, float],
    config: CandidateGraphConfig,
) -> SparseCandidateGraph:
    """Create sparse continuation/gap edges and paired division hypotheses."""
    nodes = sorted(detections, key=lambda item: (item.t, -item.score, item.z, item.y, item.x))
    by_t: dict[int, list[int]] = {}
    for index, node in enumerate(nodes):
        by_t.setdefault(node.t, []).append(index)

    spacing = np.asarray(voxel_spacing_zyx, dtype=np.float64)
    edges: list[CandidateEdge] = []
    one_step_targets: dict[int, list[int]] = {}
    for source_index, source in enumerate(nodes):
        for delta_t in range(1, config.max_gap + 1):
            candidates: list[tuple[float, int]] = []
            for target_index in by_t.get(source.t + delta_t, []):
                target = nodes[target_index]
                distance = physical_distance_zyx(
                    (source.z, source.y, source.x),
                    (target.z, target.y, target.x),
                    voxel_spacing_zyx,
                )
                if distance <= config.max_speed_um_per_frame * delta_t:
                    candidates.append((distance, target_index))
            candidates.sort(key=lambda item: (item[0], item[1]))
            for distance, target_index in candidates[: config.max_neighbors]:
                target = nodes[target_index]
                displacement = (
                    np.asarray((target.z, target.y, target.x), dtype=np.float64)
                    - np.asarray((source.z, source.y, source.x), dtype=np.float64)
                ) * spacing
                edges.append(
                    CandidateEdge(
                        source=source_index,
                        target=target_index,
                        delta_t=delta_t,
                        distance_um=distance,
                        confidence_mean=float((source.score + target.score) / 2),
                        displacement_zyx_um=(
                            float(displacement[0]),
                            float(displacement[1]),
                            float(displacement[2]),
                        ),
                        appearance_similarity=_appearance_similarity(
                            source.appearance_embedding, target.appearance_embedding
                        ),
                        intensity_log_ratio=_optional_log_ratio(source.intensity, target.intensity),
                        volume_log_ratio=_optional_log_ratio(source.volume, target.volume),
                        temporal_density_log_ratio=float(
                            abs(
                                np.log(
                                    (len(by_t.get(target.t, [])) + 1)
                                    / (len(by_t.get(source.t, [])) + 1)
                                )
                            )
                        ),
                    )
                )
                if delta_t == 1:
                    one_step_targets.setdefault(source_index, []).append(target_index)

    divisions: list[DivisionCandidate] = []
    if config.divisions_enabled:
        for source_index, targets in one_step_targets.items():
            source = nodes[source_index]
            closest = sorted(
                set(targets),
                key=lambda index: physical_distance_zyx(
                    (source.z, source.y, source.x),
                    (nodes[index].z, nodes[index].y, nodes[index].x),
                    voxel_spacing_zyx,
                ),
            )[: config.max_division_children]
            for child_a, child_b in combinations(closest, 2):
                first, second = nodes[child_a], nodes[child_b]
                separation = physical_distance_zyx(
                    (first.z, first.y, first.x),
                    (second.z, second.y, second.x),
                    voxel_spacing_zyx,
                )
                if separation > config.max_daughter_separation_um:
                    continue
                distance_a = physical_distance_zyx(
                    (source.z, source.y, source.x),
                    (first.z, first.y, first.x),
                    voxel_spacing_zyx,
                )
                distance_b = physical_distance_zyx(
                    (source.z, source.y, source.x),
                    (second.z, second.y, second.x),
                    voxel_spacing_zyx,
                )
                midpoint = (
                    np.asarray((first.z, first.y, first.x), dtype=np.float64)
                    + np.asarray((second.z, second.y, second.x), dtype=np.float64)
                ) / 2
                midpoint_zyx = (
                    float(midpoint[0]),
                    float(midpoint[1]),
                    float(midpoint[2]),
                )
                midpoint_distance = physical_distance_zyx(
                    (source.z, source.y, source.x), midpoint_zyx, voxel_spacing_zyx
                )
                divisions.append(
                    DivisionCandidate(
                        source=source_index,
                        child_a=child_a,
                        child_b=child_b,
                        parent_child_distance_mean_um=(distance_a + distance_b) / 2,
                        daughter_separation_um=separation,
                        midpoint_distance_um=midpoint_distance,
                        confidence_mean=float((source.score + first.score + second.score) / 3),
                    )
                )
    return SparseCandidateGraph(nodes=nodes, edges=edges, divisions=divisions)
