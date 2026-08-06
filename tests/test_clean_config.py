from __future__ import annotations

import copy

import pytest
import yaml

from biohub_pipeline.config import ConfigError, load_config, validate_config


def test_clean_v106_upstream_defaults() -> None:
    config = load_config("configs/clean_v106.yaml")
    assert config.source["version"] == 106
    assert config.source["license"] == "Apache-2.0"
    assert config.inference["detection_threshold"] == 0.96875
    assert config.inference["ilp_appearance_weight"] == 0.0
    assert config.inference["ilp_disappearance_weight"] == 1.575
    assert config.postprocessing["motion_relink_relaxed_um"] == 9.5
    assert config.postprocessing["short_track_rescue_min_mean_edge_prob"] == 0.90
    assert config.postprocessing["short_track_rescue_max_nodes_abs"] == 60
    assert config.local_cv["fixed_dataset_count"] == 8


def test_invalid_config_is_rejected() -> None:
    with open("configs/clean_v106.yaml", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    invalid = copy.deepcopy(raw)
    invalid["inference"]["detection_threshold"] = 2.0
    with pytest.raises(ConfigError, match="detection_threshold"):
        validate_config(invalid)
