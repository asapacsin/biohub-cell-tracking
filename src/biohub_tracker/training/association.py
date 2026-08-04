from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from biohub_tracker.annotation_reader import read_geff_graph
from biohub_tracker.association.candidates import CandidateGraphConfig, build_candidate_graph
from biohub_tracker.association.learned import (
    LinearAssociationArtifact,
    candidate_feature_matrix,
    fit_linear_association_model,
)
from biohub_tracker.models import DetectionCandidate
from biohub_tracker.training.data import discover_training_pairs
from biohub_tracker.training.labels import label_candidate_graph
from biohub_tracker.zarr_reader import VolumeDatasetReader


@dataclass(frozen=True, slots=True)
class AssociationTrainingConfig:
    candidate_graph: CandidateGraphConfig = field(default_factory=CandidateGraphConfig)
    l2_regularization: float = 1e-3
    max_iterations: int = 500


def train_association_model(
    competition_root: str | Path,
    output_path: str | Path,
    config: AssociationTrainingConfig,
) -> LinearAssociationArtifact:
    """Generate GEFF candidates, fit the event scorer, and save a portable artifact."""
    reader = VolumeDatasetReader(competition_root, split="train")
    feature_batches: list[np.ndarray] = []
    label_batches: list[np.ndarray] = []
    for pair in discover_training_pairs(competition_root):
        lineage = read_geff_graph(pair.lineage_store)
        detections = [
            DetectionCandidate(
                dataset=pair.dataset,
                t=node.t,
                z=node.z,
                y=node.y,
                x=node.x,
                score=1.0,
                annotation_id=node.node_id,
            )
            for node in lineage.nodes
        ]
        graph = build_candidate_graph(
            detections,
            voxel_spacing_zyx=reader.metadata(pair.dataset).voxel_spacing_zyx,
            config=config.candidate_graph,
        )
        detection_to_geff = {
            index: node.annotation_id
            for index, node in enumerate(graph.nodes)
            if node.annotation_id is not None
        }
        candidate_labels = label_candidate_graph(
            graph,
            detection_to_geff_node=detection_to_geff,
            geff_edges=set(lineage.edges),
        )
        feature_batches.append(candidate_feature_matrix(graph))
        label_batches.append(
            np.concatenate((candidate_labels.edge_labels, candidate_labels.division_labels)).astype(
                np.int8
            )
        )
    features = np.concatenate(feature_batches)
    all_labels = np.concatenate(label_batches)
    known = all_labels >= 0
    artifact = fit_linear_association_model(
        features[known],
        all_labels[known],
        l2_regularization=config.l2_regularization,
        max_iterations=config.max_iterations,
    )
    artifact.save(output_path)
    return artifact
