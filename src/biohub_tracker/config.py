from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from biohub_tracker.association import (
    AssociationScoringConfig,
    CandidateGraphConfig,
    OptimizerConfig,
)
from biohub_tracker.detection import BlobDetectionConfig, DetectionDecoderConfig
from biohub_tracker.postprocessing import PostprocessingConfig
from biohub_tracker.preprocessing import PreprocessingConfig
from biohub_tracker.tracking import TrackingConfig


@dataclass(frozen=True, slots=True)
class SubmissionConfig:
    node_id_start: int = 1
    sort_rows: bool = True
    strict_validation: bool = True


@dataclass(frozen=True, slots=True)
class LearnedDetectorConfig:
    model_paths: tuple[str, ...] = ()
    device: str = "cpu"
    tta_flips: bool = True
    window_shape_zyx: tuple[int, int, int] = (32, 128, 128)
    window_overlap: float = 0.25

    def __post_init__(self) -> None:
        if any(size < 1 for size in self.window_shape_zyx):
            raise ValueError("detection.window_shape_zyx must contain positive sizes")
        if not 0 <= self.window_overlap < 1:
            raise ValueError("detection.window_overlap must be in [0, 1)")


@dataclass(frozen=True, slots=True)
class ProjectConfig:
    raw: dict[str, Any]
    submission: SubmissionConfig
    preprocessing: PreprocessingConfig
    detection_method: str
    detection: BlobDetectionConfig
    learned_detection: LearnedDetectorConfig
    decoder: DetectionDecoderConfig
    candidate_graph: CandidateGraphConfig
    scoring: AssociationScoringConfig
    association_scorer_kind: str
    association_model_path: str | None
    optimizer: OptimizerConfig
    postprocessing: PostprocessingConfig
    tracking: TrackingConfig


def _mapping(raw: dict[str, Any], key: str) -> dict[str, Any]:
    value = raw.get(key, {})
    if not isinstance(value, dict):
        raise ValueError(f"{key} configuration must be a mapping")
    return value


def _int_tuple3(value: Any, name: str) -> tuple[int, int, int]:
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"{name} must be a three-item list")
    return int(value[0]), int(value[1]), int(value[2])


def load_config(path: str | Path) -> ProjectConfig:
    source = Path(path)
    with source.open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Configuration root must be a mapping: {source}")

    submission_raw = _mapping(raw, "submission")
    submission = SubmissionConfig(
        node_id_start=int(submission_raw.get("node_id_start", 1)),
        sort_rows=bool(submission_raw.get("sort_rows", True)),
        strict_validation=bool(submission_raw.get("strict_validation", True)),
    )
    if submission.node_id_start < 0:
        raise ValueError("submission.node_id_start must be non-negative")

    preprocessing_raw = _mapping(raw, "preprocessing")
    preprocessing = PreprocessingConfig(
        lower_percentile=float(preprocessing_raw.get("lower_percentile", 1.0)),
        upper_percentile=float(preprocessing_raw.get("upper_percentile", 99.8)),
        clip=bool(preprocessing_raw.get("clip", True)),
    )

    detection_raw = _mapping(raw, "detection")
    method = str(detection_raw.get("method", "blob")).lower()
    if method not in {"blob", "learned"}:
        raise ValueError(f"Unsupported detection.method {method!r}; use 'blob' or 'learned'")
    detection = BlobDetectionConfig(
        lower_percentile=float(detection_raw.get("lower_percentile", 1.0)),
        upper_percentile=float(detection_raw.get("upper_percentile", 99.8)),
        gaussian_sigma_um=float(detection_raw.get("gaussian_sigma_um", 1.5)),
        threshold=float(detection_raw.get("threshold", 0.08)),
        minimum_separation_um=float(detection_raw.get("minimum_separation_um", 4.0)),
    )
    paths = detection_raw.get("model_paths", [])
    if isinstance(paths, str):
        paths = [paths]
    if not isinstance(paths, list):
        raise ValueError("detection.model_paths must be a list of TorchScript artifacts")
    learned_detection = LearnedDetectorConfig(
        model_paths=tuple(str(value) for value in paths),
        device=str(detection_raw.get("device", "cpu")),
        tta_flips=bool(detection_raw.get("tta_flips", True)),
        window_shape_zyx=_int_tuple3(
            detection_raw.get("window_shape_zyx", [32, 128, 128]),
            "detection.window_shape_zyx",
        ),
        window_overlap=float(detection_raw.get("window_overlap", 0.25)),
    )
    decoder_raw = detection_raw.get("decoder", {})
    if not isinstance(decoder_raw, dict):
        raise ValueError("detection.decoder configuration must be a mapping")
    decoder = DetectionDecoderConfig(
        threshold=float(decoder_raw.get("threshold", 0.35)),
        adaptive_quantile=float(decoder_raw.get("adaptive_quantile", 0.995)),
        nms_radius_um=float(decoder_raw.get("nms_radius_um", 3.0)),
        refinement_radius_voxels=int(decoder_raw.get("refinement_radius_voxels", 1)),
    )

    association_raw = _mapping(raw, "association")
    candidate_raw = association_raw.get("candidate_graph", {})
    scorer_raw = association_raw.get("scorer", {})
    optimizer_raw = association_raw.get("optimizer", {})
    for name, value in (
        ("association.candidate_graph", candidate_raw),
        ("association.scorer", scorer_raw),
        ("association.optimizer", optimizer_raw),
    ):
        if not isinstance(value, dict):
            raise ValueError(f"{name} configuration must be a mapping")
    candidate_graph = CandidateGraphConfig(
        max_neighbors=int(candidate_raw.get("max_neighbors", 8)),
        max_gap=int(candidate_raw.get("max_gap", 2)),
        max_speed_um_per_frame=float(candidate_raw.get("max_speed_um_per_frame", 15.0)),
        divisions_enabled=bool(candidate_raw.get("divisions_enabled", True)),
        max_division_children=int(candidate_raw.get("max_division_children", 6)),
        max_daughter_separation_um=float(candidate_raw.get("max_daughter_separation_um", 16.0)),
    )
    scoring = AssociationScoringConfig(
        link_bias=float(scorer_raw.get("link_bias", 15.0)),
        distance_weight=float(scorer_raw.get("distance_weight", 1.0)),
        confidence_weight=float(scorer_raw.get("confidence_weight", 4.0)),
        gap_penalty=float(scorer_raw.get("gap_penalty", 2.0)),
        appearance_weight=float(scorer_raw.get("appearance_weight", 0.0)),
        intensity_weight=float(scorer_raw.get("intensity_weight", 0.0)),
        volume_weight=float(scorer_raw.get("volume_weight", 0.0)),
        temporal_context_weight=float(scorer_raw.get("temporal_context_weight", 0.0)),
        division_bias=float(scorer_raw.get("division_bias", 18.0)),
        division_midpoint_weight=float(scorer_raw.get("division_midpoint_weight", 1.5)),
        division_separation_weight=float(scorer_raw.get("division_separation_weight", 0.2)),
    )
    scorer_kind = str(scorer_raw.get("kind", "handcrafted")).lower()
    if scorer_kind not in {"handcrafted", "learned_linear"}:
        raise ValueError("association.scorer.kind must be 'handcrafted' or 'learned_linear'")
    model_path_value = scorer_raw.get("model_path")
    association_model_path = str(model_path_value) if model_path_value is not None else None
    optimizer = OptimizerConfig(
        method=str(optimizer_raw.get("method", "ilp")),
        minimum_score=float(optimizer_raw.get("minimum_score", 0.0)),
        time_limit_seconds=float(optimizer_raw.get("time_limit_seconds", 120.0)),
    )

    post_raw = _mapping(raw, "postprocessing")
    postprocessing = PostprocessingConfig(
        minimum_track_length=int(post_raw.get("minimum_track_length", 1)),
        remove_isolated_short_tracks=bool(post_raw.get("remove_isolated_short_tracks", False)),
    )

    # Retained for callers of the older consecutive-frame linker.
    tracking_raw = _mapping(raw, "tracking")
    tracking = TrackingConfig(
        candidate_neighbors=int(tracking_raw.get("candidate_neighbors", 5)),
        max_match_distance_um=float(tracking_raw.get("max_match_distance_um", 15.0)),
        use_hungarian=bool(tracking_raw.get("use_hungarian", True)),
        distance_weight=float(tracking_raw.get("distance_weight", 1.0)),
        intensity_weight=float(tracking_raw.get("intensity_weight", 0.0)),
        volume_weight=float(tracking_raw.get("volume_weight", 0.0)),
    )
    return ProjectConfig(
        raw=raw,
        submission=submission,
        preprocessing=preprocessing,
        detection_method=method,
        detection=detection,
        learned_detection=learned_detection,
        decoder=decoder,
        candidate_graph=candidate_graph,
        scoring=scoring,
        association_scorer_kind=scorer_kind,
        association_model_path=association_model_path,
        optimizer=optimizer,
        postprocessing=postprocessing,
        tracking=tracking,
    )
