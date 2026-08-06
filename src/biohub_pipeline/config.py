"""Configuration loading for the selected clean V106 pipeline.

SPDX-License-Identifier: Apache-2.0
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when a clean-pipeline configuration is invalid."""


@dataclass(frozen=True)
class PipelineConfig:
    source: dict[str, Any]
    inference: dict[str, Any]
    postprocessing: dict[str, Any]
    local_cv: dict[str, Any]


_SECTIONS = {"source", "inference", "postprocessing", "local_cv"}
_INFERENCE_REQUIRED = {
    "method": str,
    "weights_relative": str,
    "support_artifact": str,
    "prediction_script": str,
    "detection_threshold": (int, float),
    "unet_batch_size": int,
    "use_ilp": bool,
    "ilp_edge_weight": (int, float),
    "ilp_appearance_weight": (int, float),
    "ilp_disappearance_weight": (int, float),
    "ilp_division_weight": (int, float),
    "spatial_d4_tta": bool,
}


def _require_mapping(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigError(f"{name} must be a mapping")
    return value


def _require_keys(section: dict[str, Any], expected: dict[str, object], name: str) -> None:
    missing = sorted(set(expected) - set(section))
    if missing:
        raise ConfigError(f"{name} is missing keys: {missing}")
    for key, expected_type in expected.items():
        value = section[key]
        if isinstance(value, bool) and expected_type is not bool:
            raise ConfigError(f"{name}.{key} has invalid boolean value")
        if not isinstance(value, expected_type):
            raise ConfigError(f"{name}.{key} has invalid type {type(value).__name__}")


def validate_config(raw: object) -> PipelineConfig:
    root = _require_mapping(raw, "configuration")
    unknown = sorted(set(root) - _SECTIONS)
    missing = sorted(_SECTIONS - set(root))
    if unknown:
        raise ConfigError(f"unknown configuration sections: {unknown}")
    if missing:
        raise ConfigError(f"missing configuration sections: {missing}")

    source = _require_mapping(root["source"], "source")
    inference = _require_mapping(root["inference"], "inference")
    postprocessing = _require_mapping(root["postprocessing"], "postprocessing")
    local_cv = _require_mapping(root["local_cv"], "local_cv")
    _require_keys(inference, _INFERENCE_REQUIRED, "inference")

    if source.get("version") != 106:
        raise ConfigError("source.version must identify the selected Version 106 snapshot")
    if source.get("license") != "Apache-2.0":
        raise ConfigError("source.license must remain Apache-2.0")
    if not 0.0 < float(inference["detection_threshold"]) <= 1.0:
        raise ConfigError("inference.detection_threshold must be in (0, 1]")
    if int(inference["unet_batch_size"]) < 1:
        raise ConfigError("inference.unet_batch_size must be positive")

    scale = postprocessing.get("voxel_scale_um")
    if not isinstance(scale, list) or len(scale) != 3 or any(float(v) <= 0 for v in scale):
        raise ConfigError("postprocessing.voxel_scale_um must contain three positive values")
    if int(postprocessing.get("output_min_track_len", 0)) < 1:
        raise ConfigError("postprocessing.output_min_track_len must be positive")
    if int(postprocessing.get("short_track_rescue_min_len", 0)) < 1:
        raise ConfigError("postprocessing.short_track_rescue_min_len must be positive")
    if int(local_cv.get("fixed_dataset_count", 0)) != 8:
        raise ConfigError("local_cv.fixed_dataset_count must preserve the upstream fixed-8 design")

    return PipelineConfig(source, inference, postprocessing, local_cv)


def load_config(path: str | Path) -> PipelineConfig:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"configuration file not found: {source}")
    return validate_config(yaml.safe_load(source.read_text(encoding="utf-8")))
