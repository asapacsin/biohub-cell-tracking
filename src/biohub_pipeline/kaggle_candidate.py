"""Prepare and launch the candidate-edge bottleneck experiment on Kaggle."""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from biohub_pipeline.fixed8_cv import FIXED8_DATASETS

COMPETITION_SLUG = "biohub-cell-tracking-during-development"
SUPPORT_OWNER = "pilkwang"
SUPPORT_SLUG = "biohub-tracking-support-pack-50ep-v1"
SEED2_OWNER = "pilkwang"
SEED2_SLUG = "biohub-temporal-unet3d-seed314159-v1"
PRIMARY_CHECKPOINT_SHA256 = "12f6881ee3620a831697ca098ff8f48e687a24225f4e048b538deec3562fe771"
SECONDARY_CHECKPOINT_SHA256 = "9bac2fa0dadc4a6fc1899e0caf187f4b553e0a7cd90ba1261a68b35ffe9e305f"


@dataclass(frozen=True)
class PreparedInputs:
    data_dir: Path
    support_dir: Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def locate_competition_root(input_root: Path) -> Path:
    """Resolve current and legacy Kaggle competition mount layouts."""
    candidates = (
        input_root / "competitions" / COMPETITION_SLUG,
        input_root / COMPETITION_SLUG,
    )
    for candidate in candidates:
        if (candidate / "train").is_dir():
            return candidate
    rendered = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"competition train mount not found; checked: {rendered}")


def locate_dataset_root(input_root: Path, owner: str, slug: str) -> Path:
    """Resolve current and legacy Kaggle dataset mount layouts."""
    candidates = (
        input_root / "datasets" / owner / slug,
        input_root / slug,
    )
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    rendered = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"dataset mount {owner}/{slug} not found; checked: {rendered}")


def resolve_checkpoint(root: Path, expected_sha256: str) -> Path:
    """Find exactly one checkpoint matching the pinned provenance hash."""
    matches = [
        path
        for path in sorted(root.rglob("edge_predictor_best.pth"))
        if _sha256(path) == expected_sha256
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"expected exactly one checkpoint with SHA-256 {expected_sha256}, found {len(matches)}"
        )
    return matches[0]


def fixed8_sources(competition_root: Path) -> dict[str, tuple[Path, Path]]:
    """Validate the exact fixed-eight training image/ground-truth pairs."""
    train = competition_root / "train"
    result: dict[str, tuple[Path, Path]] = {}
    missing: list[str] = []
    for dataset in FIXED8_DATASETS:
        image = train / f"{dataset}.zarr"
        ground_truth = train / f"{dataset}.geff"
        if not image.is_dir():
            missing.append(str(image))
        if not ground_truth.is_dir():
            missing.append(str(ground_truth))
        result[dataset] = (image, ground_truth)
    if missing:
        raise FileNotFoundError("missing fixed-eight inputs:\n" + "\n".join(missing))
    return result


def _symlink(source: Path, target: Path, *, directory: bool) -> None:
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"staging target already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.symlink_to(source.resolve(), target_is_directory=directory)


def prepare_inputs(input_root: Path, workspace: Path) -> PreparedInputs:
    """Stage lightweight symlinks without copying competition data or weights."""
    if workspace.exists():
        raise FileExistsError(f"workspace already exists: {workspace}")

    competition_root = locate_competition_root(input_root)
    support_root = locate_dataset_root(input_root, SUPPORT_OWNER, SUPPORT_SLUG)
    seed2_root = locate_dataset_root(input_root, SEED2_OWNER, SEED2_SLUG)
    primary = resolve_checkpoint(support_root, PRIMARY_CHECKPOINT_SHA256)
    secondary = resolve_checkpoint(seed2_root, SECONDARY_CHECKPOINT_SHA256)

    data_dir = workspace / "fixed8_data"
    support_dir = workspace / "support"
    data_dir.mkdir(parents=True)
    support_dir.mkdir(parents=True)

    for dataset, (image, ground_truth) in fixed8_sources(competition_root).items():
        _symlink(image, data_dir / f"{dataset}.zarr", directory=True)
        _symlink(ground_truth, data_dir / f"{dataset}.geff", directory=True)

    repo = support_root / "repo"
    wheels = support_root / "wheels"
    if not (repo / "scripts" / "predict_unet_transformer.py").is_file():
        raise FileNotFoundError(f"support predictor is missing under {repo}")
    if not wheels.is_dir():
        raise FileNotFoundError(f"support wheels are missing: {wheels}")
    _symlink(repo, support_dir / "repo", directory=True)
    _symlink(wheels, support_dir / "wheels", directory=True)
    _symlink(
        primary,
        support_dir / "weights/unet_transformer/split_0/edge_predictor_best.pth",
        directory=False,
    )
    _symlink(
        secondary,
        support_dir / "weights/unet_transformer/seed_314159/edge_predictor_best.pth",
        directory=False,
    )
    return PreparedInputs(data_dir=data_dir, support_dir=support_dir)


def cuda_available() -> bool:
    """Run both required CUDA probes and report their results."""
    try:
        nvidia = subprocess.run(
            ["nvidia-smi"],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        print("nvidia-smi: command not found")
        nvidia_ok = False
    else:
        print(nvidia.stdout or nvidia.stderr)
        nvidia_ok = nvidia.returncode == 0

    try:
        import torch
    except ImportError:
        print("torch.cuda.is_available(): torch is not installed")
        return False
    torch_ok = bool(torch.cuda.is_available())
    print(f"torch.cuda.is_available(): {torch_ok}")
    if torch_ok:
        print(f"torch CUDA device: {torch.cuda.get_device_name(0)}")
    return nvidia_ok and torch_ok


def experiment_command(
    repo_root: Path,
    prepared: PreparedInputs,
    output_dir: Path,
    work_dir: Path,
    top_k: int,
) -> list[str]:
    if top_k < 2:
        raise ValueError("top_k must be at least 2")
    return [
        sys.executable,
        str(repo_root / "scripts/run_candidate_bottleneck_experiment.py"),
        "--data-dir",
        str(prepared.data_dir),
        "--support-dir",
        str(prepared.support_dir),
        "--config",
        str(repo_root / "configs/clean_v106_two_seed.yaml"),
        "--output-dir",
        str(output_dir),
        "--work-dir",
        str(work_dir),
        "--top-k",
        str(top_k),
    ]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=Path("/kaggle/input"))
    parser.add_argument("--workspace", type=Path, default=Path("/kaggle/working/biohub-candidate"))
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--top-k", type=int, default=16)
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Stage inputs and print the command without launching GPU inference.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    repo_root = args.repo_root.resolve()
    workspace = args.workspace.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else workspace.parent / "candidate_edge_bottleneck_v1"
    )
    gpu_ready = cuda_available()
    prepared = prepare_inputs(args.input_root.resolve(), workspace)
    command = experiment_command(
        repo_root,
        prepared,
        output_dir,
        workspace / "work",
        args.top_k,
    )
    print("Prepared experiment command:")
    print(subprocess.list2cmdline(command))
    if args.prepare_only:
        return 0
    if not gpu_ready:
        raise RuntimeError(
            "CUDA is unavailable; preparation completed but inference was not started"
        )
    subprocess.run(command, cwd=repo_root, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
