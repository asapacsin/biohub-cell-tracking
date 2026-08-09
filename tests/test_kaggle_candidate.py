from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from biohub_pipeline.fixed8_cv import FIXED8_DATASETS
from biohub_pipeline.kaggle_candidate import (
    PreparedInputs,
    experiment_command,
    fixed8_sources,
    locate_competition_root,
    locate_dataset_root,
    prepare_inputs,
    resolve_checkpoint,
)


def _checkpoint(path: Path, content: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return hashlib.sha256(content).hexdigest()


@pytest.mark.parametrize("modern", [True, False])
def test_locate_kaggle_mount_layouts(tmp_path: Path, modern: bool) -> None:
    competition = (
        tmp_path / "competitions/biohub-cell-tracking-during-development"
        if modern
        else tmp_path / "biohub-cell-tracking-during-development"
    )
    dataset = (
        tmp_path / "datasets/pilkwang/example"
        if modern
        else tmp_path / "example"
    )
    (competition / "train").mkdir(parents=True)
    dataset.mkdir(parents=True)

    assert locate_competition_root(tmp_path) == competition
    assert locate_dataset_root(tmp_path, "pilkwang", "example") == dataset


def test_fixed8_sources_requires_every_pair(tmp_path: Path) -> None:
    train = tmp_path / "train"
    train.mkdir()
    for dataset in FIXED8_DATASETS:
        (train / f"{dataset}.zarr").mkdir()
        (train / f"{dataset}.geff").mkdir()

    sources = fixed8_sources(tmp_path)

    assert list(sources) == FIXED8_DATASETS
    (train / f"{FIXED8_DATASETS[-1]}.geff").rmdir()
    with pytest.raises(FileNotFoundError, match="missing fixed-eight inputs"):
        fixed8_sources(tmp_path)


def test_resolve_checkpoint_uses_pinned_content_hash(tmp_path: Path) -> None:
    _checkpoint(tmp_path / "wrong/edge_predictor_best.pth", b"wrong")
    expected = _checkpoint(tmp_path / "right/edge_predictor_best.pth", b"right")

    assert resolve_checkpoint(tmp_path, expected) == tmp_path / "right/edge_predictor_best.pth"


def test_prepare_inputs_stages_symlinks_without_copying(tmp_path: Path, monkeypatch) -> None:
    input_root = tmp_path / "input"
    competition = input_root / "competitions/biohub-cell-tracking-during-development"
    support = input_root / "datasets/pilkwang/biohub-tracking-support-pack-50ep-v1"
    seed2 = input_root / "datasets/pilkwang/biohub-temporal-unet3d-seed314159-v1"
    train = competition / "train"
    for dataset in FIXED8_DATASETS:
        (train / f"{dataset}.zarr").mkdir(parents=True)
        (train / f"{dataset}.geff").mkdir()
    (support / "repo/scripts").mkdir(parents=True)
    (support / "repo/scripts/predict_unet_transformer.py").write_text("# fixture\n")
    (support / "wheels").mkdir()
    primary = support / "weights/unet_transformer/split_0/edge_predictor_best.pth"
    secondary = seed2 / "edge_predictor_best.pth"
    primary_hash = _checkpoint(primary, b"primary")
    secondary_hash = _checkpoint(secondary, b"secondary")
    monkeypatch.setattr(
        "biohub_pipeline.kaggle_candidate.PRIMARY_CHECKPOINT_SHA256", primary_hash
    )
    monkeypatch.setattr(
        "biohub_pipeline.kaggle_candidate.SECONDARY_CHECKPOINT_SHA256", secondary_hash
    )

    prepared = prepare_inputs(input_root, tmp_path / "workspace")

    assert prepared.data_dir.joinpath(f"{FIXED8_DATASETS[0]}.zarr").is_symlink()
    assert prepared.support_dir.joinpath("repo").is_symlink()
    assert prepared.support_dir.joinpath(
        "weights/unet_transformer/seed_314159/edge_predictor_best.pth"
    ).read_bytes() == b"secondary"


def test_experiment_command_is_exact_and_bounded(tmp_path: Path) -> None:
    prepared = PreparedInputs(tmp_path / "data", tmp_path / "support")

    command = experiment_command(
        tmp_path / "repo",
        prepared,
        tmp_path / "output",
        tmp_path / "work",
        16,
    )

    assert command[-2:] == ["--top-k", "16"]
    assert str(tmp_path / "repo/scripts/run_candidate_bottleneck_experiment.py") in command
    with pytest.raises(ValueError, match="at least 2"):
        experiment_command(tmp_path, prepared, tmp_path / "out", tmp_path / "work", 1)
