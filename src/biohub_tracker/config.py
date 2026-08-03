from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from biohub_tracker.detection import BlobDetectionConfig
from biohub_tracker.tracking import TrackingConfig


@dataclass(frozen=True, slots=True)
class SubmissionConfig:
    node_id_start: int = 1
    sort_rows: bool = True
    strict_validation: bool = True


@dataclass(frozen=True, slots=True)
class ProjectConfig:
    raw: dict[str, Any]
    submission: SubmissionConfig
    detection: BlobDetectionConfig
    tracking: TrackingConfig


def load_config(path: str | Path) -> ProjectConfig:
    source = Path(path)
    with source.open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Configuration root must be a mapping: {source}")
    submission_raw = raw.get("submission", {})
    if not isinstance(submission_raw, dict):
        raise ValueError("submission configuration must be a mapping")
    submission = SubmissionConfig(
        node_id_start=int(submission_raw.get("node_id_start", 1)),
        sort_rows=bool(submission_raw.get("sort_rows", True)),
        strict_validation=bool(submission_raw.get("strict_validation", True)),
    )
    if submission.node_id_start < 0:
        raise ValueError("submission.node_id_start must be non-negative")

    detection_raw = raw.get("detection", {})
    if not isinstance(detection_raw, dict):
        raise ValueError("detection configuration must be a mapping")
    method = str(detection_raw.get("method", "blob")).lower()
    if method != "blob":
        raise ValueError(f"Unsupported detection.method {method!r}; only 'blob' is available")
    detection = BlobDetectionConfig(
        lower_percentile=float(detection_raw.get("lower_percentile", 1.0)),
        upper_percentile=float(detection_raw.get("upper_percentile", 99.8)),
        gaussian_sigma_um=float(detection_raw.get("gaussian_sigma_um", 1.5)),
        threshold=float(detection_raw.get("threshold", 0.08)),
        minimum_separation_um=float(detection_raw.get("minimum_separation_um", 4.0)),
    )

    tracking_raw = raw.get("tracking", {})
    if not isinstance(tracking_raw, dict):
        raise ValueError("tracking configuration must be a mapping")
    tracking = TrackingConfig(
        candidate_neighbors=int(tracking_raw.get("candidate_neighbors", 5)),
        max_match_distance_um=float(tracking_raw.get("max_match_distance_um", 15.0)),
        use_hungarian=bool(tracking_raw.get("use_hungarian", True)),
        distance_weight=float(tracking_raw.get("distance_weight", 1.0)),
        intensity_weight=float(tracking_raw.get("intensity_weight", 0.0)),
        volume_weight=float(tracking_raw.get("volume_weight", 0.0)),
    )
    return ProjectConfig(raw=raw, submission=submission, detection=detection, tracking=tracking)
