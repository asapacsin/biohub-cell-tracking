from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from biohub_tracker.association import CandidateGraphConfig
from biohub_tracker.training.association import AssociationTrainingConfig
from biohub_tracker.training.detector import DetectorTrainingConfig, UNet3DConfig


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    detector: DetectorTrainingConfig
    detector_seeds: tuple[int, ...]
    association: AssociationTrainingConfig


def _tuple3(raw: Any, *, name: str) -> tuple[int, int, int]:
    if not isinstance(raw, list) or len(raw) != 3:
        raise ValueError(f"{name} must be a three-item list")
    return int(raw[0]), int(raw[1]), int(raw[2])


def load_training_config(path: str | Path) -> TrainingConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError("training configuration root must be a mapping")
    training = raw.get("training", {})
    if not isinstance(training, dict):
        raise ValueError("training configuration must be a mapping")
    detector_raw = training.get("detector", {})
    association_raw = training.get("association", {})
    if not isinstance(detector_raw, dict) or not isinstance(association_raw, dict):
        raise ValueError("training.detector and training.association must be mappings")
    unet_raw = detector_raw.get("unet", {})
    if not isinstance(unet_raw, dict):
        raise ValueError("training.detector.unet must be a mapping")
    detector = DetectorTrainingConfig(
        patch_shape_zyx=_tuple3(
            detector_raw.get("patch_shape_zyx", [32, 128, 128]),
            name="training.detector.patch_shape_zyx",
        ),
        sigma_um=float(detector_raw.get("sigma_um", 2.0)),
        jitter_voxels_zyx=_tuple3(
            detector_raw.get("jitter_voxels_zyx", [2, 8, 8]),
            name="training.detector.jitter_voxels_zyx",
        ),
        epochs=int(detector_raw.get("epochs", 20)),
        batch_size=int(detector_raw.get("batch_size", 2)),
        learning_rate=float(detector_raw.get("learning_rate", 1e-3)),
        validation_fraction=float(detector_raw.get("validation_fraction", 0.1)),
        positive_weight=float(detector_raw.get("positive_weight", 20.0)),
        num_workers=int(detector_raw.get("num_workers", 0)),
        seed=int(detector_raw.get("seed", 42)),
        device=str(detector_raw.get("device", "cuda")),
        unet=UNet3DConfig(
            in_channels=int(unet_raw.get("in_channels", 1)),
            out_channels=int(unet_raw.get("out_channels", 1)),
            base_channels=int(unet_raw.get("base_channels", 16)),
            depth=int(unet_raw.get("depth", 3)),
        ),
    )
    seeds_raw = detector_raw.get("seeds", [detector.seed])
    if not isinstance(seeds_raw, list) or not seeds_raw:
        raise ValueError("training.detector.seeds must be a non-empty list")
    detector_seeds = tuple(int(value) for value in seeds_raw)
    if len(set(detector_seeds)) != len(detector_seeds):
        raise ValueError("training.detector.seeds must not contain duplicates")
    candidate_raw = association_raw.get("candidate_graph", {})
    if not isinstance(candidate_raw, dict):
        raise ValueError("training.association.candidate_graph must be a mapping")
    association = AssociationTrainingConfig(
        candidate_graph=CandidateGraphConfig(
            max_neighbors=int(candidate_raw.get("max_neighbors", 8)),
            max_gap=int(candidate_raw.get("max_gap", 2)),
            max_speed_um_per_frame=float(candidate_raw.get("max_speed_um_per_frame", 15.0)),
            divisions_enabled=bool(candidate_raw.get("divisions_enabled", True)),
            max_division_children=int(candidate_raw.get("max_division_children", 6)),
            max_daughter_separation_um=float(candidate_raw.get("max_daughter_separation_um", 16.0)),
        ),
        l2_regularization=float(association_raw.get("l2_regularization", 1e-3)),
        max_iterations=int(association_raw.get("max_iterations", 500)),
    )
    return TrainingConfig(
        detector=detector,
        detector_seeds=detector_seeds,
        association=association,
    )
