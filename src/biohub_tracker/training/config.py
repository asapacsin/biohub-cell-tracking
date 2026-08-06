from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from biohub_tracker.association import CandidateGraphConfig
from biohub_tracker.training.association import AssociationTrainingConfig
from biohub_tracker.training.data import AugmentationConfig, PatchMixConfig
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


def _tuple2_float(raw: Any, *, name: str) -> tuple[float, float]:
    if not isinstance(raw, list) or len(raw) != 2:
        raise ValueError(f"{name} must be a two-item list")
    return float(raw[0]), float(raw[1])


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
    mix_raw = detector_raw.get("patch_mix", {})
    if mix_raw is None:
        mix_raw = {}
    if not isinstance(mix_raw, dict):
        raise ValueError("training.detector.patch_mix must be a mapping")
    aug_raw = detector_raw.get("augmentation", {})
    if aug_raw is None:
        aug_raw = {}
    if not isinstance(aug_raw, dict):
        raise ValueError("training.detector.augmentation must be a mapping")
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
        frame_cache_size=int(detector_raw.get("frame_cache_size", 4)),
        patch_mix=PatchMixConfig(
            positive=float(mix_raw.get("positive", 0.80)),
            near_miss=float(mix_raw.get("near_miss", 0.15)),
            empty=float(mix_raw.get("empty", 0.05)),
        ),
        positive_center_radius_um=float(detector_raw.get("positive_center_radius_um", 6.0)),
        empty_exclusion_margin_um=float(detector_raw.get("empty_exclusion_margin_um", 4.0)),
        augmentation=AugmentationConfig(
            enabled=bool(aug_raw.get("enabled", True)),
            flip_prob=float(aug_raw.get("flip_prob", 0.5)),
            rot90_prob=float(aug_raw.get("rot90_prob", 0.5)),
            intensity_scale=_tuple2_float(
                aug_raw.get("intensity_scale", [0.9, 1.1]),
                name="training.detector.augmentation.intensity_scale",
            ),
            intensity_shift=_tuple2_float(
                aug_raw.get("intensity_shift", [-0.05, 0.05]),
                name="training.detector.augmentation.intensity_shift",
            ),
            noise_std=float(aug_raw.get("noise_std", 0.02)),
            blur_sigma_px=_tuple2_float(
                aug_raw.get("blur_sigma_px", [0.0, 0.8]),
                name="training.detector.augmentation.blur_sigma_px",
            ),
        ),
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
