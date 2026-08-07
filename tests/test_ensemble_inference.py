from __future__ import annotations

import copy
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

from biohub_pipeline.config import ConfigError, PipelineConfig, load_config, validate_config
from biohub_pipeline.inference import (
    apply_logit_ensemble_patch,
    blend_logits,
    build_predict_command,
)


def _ensemble_config(secondary: str | None, alpha: float = 0.5) -> PipelineConfig:
    baseline = load_config("configs/clean_v106.yaml")
    inference = dict(baseline.inference)
    inference["ensemble_weights_relative"] = secondary
    inference["ensemble_alpha"] = alpha
    return PipelineConfig(
        source=baseline.source,
        inference=inference,
        postprocessing=baseline.postprocessing,
        local_cv=baseline.local_cv,
    )


def test_single_model_command_is_unchanged(tmp_path: Path) -> None:
    config = load_config("configs/clean_v106.yaml")
    data_dir = tmp_path / "data"
    repo_dir = tmp_path / "support" / "repo"
    weights = tmp_path / "support" / Path(str(config.inference["weights_relative"]))
    repo_dir.mkdir(parents=True)

    command, _ = build_predict_command(config, data_dir, repo_dir, weights, ["video_a"])

    assert command == [
        sys.executable,
        "scripts/predict_unet_transformer.py",
        "--data-dir",
        str(data_dir),
        "--splits",
        "clean_v106_test_splits.json",
        "--split",
        "0",
        "--weights",
        os.path.relpath(weights, repo_dir),
        "--unet-batch-size",
        "4",
        "--det-threshold",
        "0.96875",
        "--ilp-edge-weight",
        "-1.0",
        "--ilp-appearance-weight",
        "0.0",
        "--ilp-disappearance-weight",
        "1.575",
        "--ilp-division-weight",
        "1.0",
        "--use-ilp",
    ]


def test_ensemble_command_contains_both_checkpoints_and_alpha(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _ensemble_config("weights/unet_transformer/seed_2/edge_predictor_best.pth")
    support = tmp_path / "support"
    repo_dir = support / "repo"
    primary = support / Path(str(config.inference["weights_relative"]))
    secondary = support / Path(str(config.inference["ensemble_weights_relative"]))
    repo_dir.mkdir(parents=True)
    primary.parent.mkdir(parents=True)
    secondary.parent.mkdir(parents=True)
    primary.write_bytes(b"independent-seed-one")
    secondary.write_bytes(b"independent-seed-two")
    patched: list[tuple[Path, str]] = []
    monkeypatch.setattr(
        "biohub_pipeline.inference.apply_logit_ensemble_patch",
        lambda repo, script: patched.append((repo, script)) or True,
    )

    command, _ = build_predict_command(config, tmp_path / "data", repo_dir, primary, ["video"])

    assert command[command.index("--weights") + 1] == os.path.relpath(primary, repo_dir)
    assert command[command.index("--ensemble-weights") + 1] == os.path.relpath(
        secondary, repo_dir
    )
    assert command[command.index("--ensemble-alpha") + 1] == "0.5"
    assert patched == [(repo_dir, "scripts/predict_unet_transformer.py")]


def test_missing_second_checkpoint_fails_clearly(tmp_path: Path) -> None:
    config = _ensemble_config("weights/unet_transformer/seed_2/edge_predictor_best.pth")
    support = tmp_path / "support"
    repo_dir = support / "repo"
    primary = support / Path(str(config.inference["weights_relative"]))
    repo_dir.mkdir(parents=True)
    primary.parent.mkdir(parents=True)
    primary.write_bytes(b"primary")

    with pytest.raises(FileNotFoundError, match="ensemble checkpoint is missing"):
        build_predict_command(config, tmp_path / "data", repo_dir, primary, ["video"])


def test_duplicate_checkpoint_content_is_rejected(tmp_path: Path) -> None:
    config = _ensemble_config("weights/unet_transformer/seed_2/edge_predictor_best.pth")
    support = tmp_path / "support"
    repo_dir = support / "repo"
    primary = support / Path(str(config.inference["weights_relative"]))
    secondary = support / Path(str(config.inference["ensemble_weights_relative"]))
    repo_dir.mkdir(parents=True)
    primary.parent.mkdir(parents=True)
    secondary.parent.mkdir(parents=True)
    primary.write_bytes(b"same-checkpoint")
    secondary.write_bytes(b"same-checkpoint")

    with pytest.raises(ValueError, match="two distinct independently trained checkpoints"):
        build_predict_command(config, tmp_path / "data", repo_dir, primary, ["video"])


def test_invalid_ensemble_alpha_is_rejected() -> None:
    with open("configs/clean_v106.yaml", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    invalid = copy.deepcopy(raw)
    invalid["inference"]["ensemble_alpha"] = 1.01

    with pytest.raises(ConfigError, match="ensemble_alpha must be between 0 and 1"):
        validate_config(invalid)


def test_blend_math_is_raw_linear_interpolation() -> None:
    seed1 = np.array([2.0, 0.0])
    seed2 = np.array([0.0, 2.0])

    np.testing.assert_array_equal(blend_logits(seed1, seed2, 0.5), np.array([1.0, 1.0]))


def test_support_patch_blends_detector_and_edge_logits_before_activation(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    script = repo / "scripts" / "predict_unet_transformer.py"
    script.parent.mkdir(parents=True)
    script.write_text(
        '''import contextlib

def load_model(
    weights_path: Path, device: torch.device,
) -> tuple[UNetNodeTransformer, int, tuple[int, ...]]:
    pass


# =============================================================================
# Per-frame loading
# =============================================================================

def detect(logits):
    return torch.sigmoid(logits) > 0.5

def predict(
    data_dir: Path,
    fold: int,
    splits_file: Path,
    weights_path: Path,
    cfg: PredictConfig,
):
    model, window_size, downsample = load_model(weights_path, device)
    edge_logits_pair = model.predict_edges(source, target)
    raw = edge_logits_pair[0]
    probs = torch.sigmoid(raw)

    parser.add_argument("--weights", type=str, default=None,
                        help="Path to weights file. "
                             "Default: weights/{method}/split_{split}/edge_predictor_best.pth")

    predict(
            weights_path=weights_path,
            cfg=cfg,
    )
''',
        encoding="utf-8",
    )

    assert apply_logit_ensemble_patch(repo, "scripts/predict_unet_transformer.py") is True
    patched = script.read_text(encoding="utf-8")

    assert "detector = [\n            _blend_raw_logits" in patched
    assert "return _blend_raw_logits(logits1, logits2, self.alpha)" in patched
    assert patched.index("edge_logits_pair = model.predict_edges") < patched.index(
        "probs = torch.sigmoid(raw)"
    )
    assert apply_logit_ensemble_patch(repo, "scripts/predict_unet_transformer.py") is False
