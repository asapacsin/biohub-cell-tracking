from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from biohub_tracker.association.candidates import SparseCandidateGraph


@dataclass(frozen=True, slots=True)
class AssociationScoringConfig:
    link_bias: float = 15.0
    distance_weight: float = 1.0
    confidence_weight: float = 4.0
    gap_penalty: float = 2.0
    appearance_weight: float = 0.0
    intensity_weight: float = 0.0
    volume_weight: float = 0.0
    temporal_context_weight: float = 0.0
    division_bias: float = 3.0
    division_midpoint_weight: float = 1.5
    division_separation_weight: float = 0.2

    def __post_init__(self) -> None:
        weights = (
            self.distance_weight,
            self.confidence_weight,
            self.gap_penalty,
            self.appearance_weight,
            self.intensity_weight,
            self.volume_weight,
            self.temporal_context_weight,
            self.division_midpoint_weight,
            self.division_separation_weight,
        )
        if any(value < 0 for value in weights):
            raise ValueError("association penalty/weight values must be non-negative")


def score_candidate_graph(
    graph: SparseCandidateGraph, config: AssociationScoringConfig
) -> SparseCandidateGraph:
    """Reference scorer; learned Trackastra/HOCT-style scorers can replace this stage."""
    for edge in graph.edges:
        appearance = edge.appearance_similarity or 0.0
        intensity_penalty = edge.intensity_log_ratio or 0.0
        volume_penalty = edge.volume_log_ratio or 0.0
        edge.score = (
            config.link_bias
            + config.confidence_weight * edge.confidence_mean
            - config.distance_weight * edge.distance_um
            - config.gap_penalty * (edge.delta_t - 1)
            + config.appearance_weight * appearance
            - config.intensity_weight * intensity_penalty
            - config.volume_weight * volume_penalty
            - config.temporal_context_weight * edge.temporal_density_log_ratio
        )
    for division in graph.divisions:
        division.score = (
            config.division_bias
            + config.confidence_weight * division.confidence_mean
            - config.distance_weight * division.parent_child_distance_mean_um
            - config.division_midpoint_weight * division.midpoint_distance_um
            - config.division_separation_weight * division.daughter_separation_um
        )
    return graph


class CandidateGraphScorer(Protocol):
    """Adapter boundary for handcrafted, Trackastra-, or HOCT-style scorers."""

    def score(self, graph: SparseCandidateGraph) -> SparseCandidateGraph: ...


@dataclass(frozen=True, slots=True)
class HandcraftedAssociationScorer:
    config: AssociationScoringConfig

    def score(self, graph: SparseCandidateGraph) -> SparseCandidateGraph:
        return score_candidate_graph(graph, self.config)
