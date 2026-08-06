"""External learned-detector/edge-scorer orchestration from the V106 notebook."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from biohub_pipeline.config import PipelineConfig


def list_stems(data_dir: Path) -> list[str]:
    stems = sorted(path.name[:-5] for path in data_dir.iterdir() if path.name.endswith(".zarr"))
    if not stems:
        raise FileNotFoundError(f"no test .zarr stores found in {data_dir}")
    return stems


def apply_spatial_d4_patch(repo_dir: Path, prediction_script: str) -> bool:
    """Apply the notebook's exact 4-way-to-D4 detector TTA source patch."""
    path = repo_dir / prediction_script
    source = path.read_text(encoding="utf-8")
    old = """        if cfg.det_tta:
            tta_flips = [(-1,), (-2,), (-2, -1)]
            for dims in tta_flips:
                imgs_flip = imgs.flip(dims)
                _, det_flip = model.encode(imgs_flip)
                for f in range(W):
                    det_logits[f] = det_logits[f] + det_flip[f].flip(dims)
                del imgs_flip, det_flip
            for f in range(W):
                det_logits[f] = det_logits[f] / 4"""
    new = """        if cfg.det_tta:
            _nv = 1
            for dims in [(-1,), (-2,), (-2, -1)]:
                imgs_flip = imgs.flip(dims)
                _, det_flip = model.encode(imgs_flip)
                for f in range(W):
                    det_logits[f] = det_logits[f] + det_flip[f].flip(dims)
                del imgs_flip, det_flip
                _nv += 1
            for _k in (1, 3):
                imgs_rot = torch.rot90(imgs, _k, dims=(-2, -1))
                _, det_rot = model.encode(imgs_rot)
                for f in range(W):
                    det_logits[f] = det_logits[f] + torch.rot90(det_rot[f], -_k, dims=(-2, -1))
                del imgs_rot, det_rot
                _nv += 1
            imgs_t = imgs.transpose(-1, -2)
            _, det_t = model.encode(imgs_t)
            for f in range(W):
                det_logits[f] = det_logits[f] + det_t[f].transpose(-1, -2)
            del imgs_t, det_t
            _nv += 1
            imgs_at = torch.rot90(imgs, 1, dims=(-2, -1)).transpose(-1, -2)
            _, det_at = model.encode(imgs_at)
            for f in range(W):
                det_logits[f] = det_logits[f] + torch.rot90(det_at[f].transpose(-1, -2), -1, dims=(-2, -1))
            del imgs_at, det_at
            _nv += 1
            for f in range(W):
                det_logits[f] = det_logits[f] / _nv"""
    if old in source:
        path.write_text(source.replace(old, new), encoding="utf-8")
        return True
    if new in source:
        return False
    raise RuntimeError("upstream detector TTA block does not match the V106 patch preimage")


def build_predict_command(
    config: PipelineConfig,
    data_dir: Path,
    repo_dir: Path,
    weights_path: Path,
    stems: list[str],
) -> tuple[list[str], Path]:
    splits = repo_dir / "clean_v106_test_splits.json"
    splits.write_text(
        json.dumps([{"split": 0, "train": [], "test": stems}], indent=2), encoding="utf-8"
    )
    inf = config.inference
    relative_weights = os.path.relpath(weights_path, repo_dir)
    command = [
        sys.executable,
        str(inf["prediction_script"]),
        "--data-dir",
        str(data_dir),
        "--splits",
        splits.name,
        "--split",
        "0",
        "--weights",
        relative_weights,
        "--unet-batch-size",
        str(inf["unet_batch_size"]),
        "--det-threshold",
        str(inf["detection_threshold"]),
        "--ilp-edge-weight",
        str(inf["ilp_edge_weight"]),
        "--ilp-appearance-weight",
        str(inf["ilp_appearance_weight"]),
        "--ilp-disappearance-weight",
        str(inf["ilp_disappearance_weight"]),
        "--ilp-division-weight",
        str(inf["ilp_division_weight"]),
    ]
    if inf["use_ilp"]:
        command.append("--use-ilp")
    return command, splits


def run_prediction(command: list[str], repo_dir: Path) -> None:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; V106 refuses CPU inference to avoid timeout")
    subprocess.run(command, cwd=repo_dir, env={**os.environ, "PYTHONPATH": "src"}, check=True)
