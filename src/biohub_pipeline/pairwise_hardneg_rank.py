"""Appearance-aware pairwise hard-negative logit reweight (pre-softmax).

score = blended_logit + w · φ(seed1, seed2, dist_um, dens)
Applied only for near-tie columns in dense scenes. Opt-in; default off.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np

FEATURE_NAMES = (
    "seed_disagree",
    "seed_min",
    "dist_n",
    "log_dens",
    "dist_x_disagree",
    "dist_x_seedmin",
)


def pairwise_dist_um(
    src_zyx: np.ndarray,
    tgt_zyx: np.ndarray,
    voxel_zyx: Sequence[float],
) -> np.ndarray:
    scale = np.asarray(voxel_zyx, dtype=np.float64).reshape(1, 1, 3)
    src = np.asarray(src_zyx, dtype=np.float64).reshape(-1, 1, 3)
    tgt = np.asarray(tgt_zyx, dtype=np.float64).reshape(1, -1, 3)
    return np.linalg.norm((src - tgt) * scale, axis=2)


def target_densities_um(
    tgt_zyx: np.ndarray,
    voxel_zyx: Sequence[float],
    radius_um: float,
) -> np.ndarray:
    scale = np.asarray(voxel_zyx, dtype=np.float64).reshape(1, 1, 3)
    pts = np.asarray(tgt_zyx, dtype=np.float64).reshape(-1, 3)
    if len(pts) == 0:
        return np.zeros(0, dtype=np.int32)
    d = np.linalg.norm((pts[:, None, :] - pts[None, :, :]) * scale, axis=2)
    return (d <= float(radius_um)).sum(axis=1).astype(np.int32)


def phi_matrix(
    seed1: np.ndarray,
    seed2: np.ndarray,
    dists: np.ndarray,
    dens: np.ndarray,
) -> np.ndarray:
    """Return (n_src, n_tgt, 6) feature tensor."""
    s1 = np.asarray(seed1, dtype=np.float64)
    s2 = np.asarray(seed2, dtype=np.float64)
    disagree = np.abs(s1 - s2)
    smin = np.minimum(s1, s2)
    dist_n = np.asarray(dists, dtype=np.float64) / 10.0
    # dens is (n_tgt,); broadcast
    log_dens = np.log1p(np.maximum(np.asarray(dens, dtype=np.float64), 0.0))
    log_dens_b = np.broadcast_to(log_dens.reshape(1, -1), dist_n.shape)
    return np.stack(
        [
            disagree,
            smin,
            dist_n,
            log_dens_b,
            dist_n * disagree,
            dist_n * smin,
        ],
        axis=-1,
    )


def apply_pairwise_hardneg_rank(
    blended_logits: np.ndarray,
    seed1_logits: np.ndarray,
    seed2_logits: np.ndarray,
    src_zyx: np.ndarray,
    tgt_zyx: np.ndarray,
    *,
    voxel_zyx: Sequence[float],
    weights: Sequence[float],
    dens_min: float,
    gap_max: float,
    radius_um: float = 15.0,
) -> tuple[np.ndarray, dict[str, int]]:
    """Return adjusted blended logits and apply counters."""
    raw = np.asarray(blended_logits, dtype=np.float64)
    if raw.ndim != 2:
        raise ValueError(f"logits must be 2D, got shape {raw.shape}")
    n_src, n_tgt = raw.shape
    if n_src == 0 or n_tgt == 0:
        return raw.copy(), {"columns": 0, "applied": 0, "skipped_single": 0}

    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    if w.shape[0] != len(FEATURE_NAMES):
        raise ValueError(f"weights must have length {len(FEATURE_NAMES)}")

    s1 = np.asarray(seed1_logits, dtype=np.float64)
    s2 = np.asarray(seed2_logits, dtype=np.float64)
    if s1.shape != raw.shape or s2.shape != raw.shape:
        raise ValueError("seed logits must match blended logits shape")

    dists = pairwise_dist_um(src_zyx, tgt_zyx, voxel_zyx)
    dens = target_densities_um(tgt_zyx, voxel_zyx, radius_um)
    feats = phi_matrix(s1, s2, dists, dens)
    correction = feats @ w
    out = raw.copy()
    applied = 0
    skipped_single = 0
    for j in range(n_tgt):
        col = out[:, j]
        if n_src < 2:
            skipped_single += 1
            continue
        top2 = np.partition(col, -2)[-2:]
        gap = float(top2.max() - top2.min())
        if gap < float(gap_max) and float(dens[j]) >= float(dens_min):
            out[:, j] = col + correction[:, j]
            applied += 1
    stats = {
        "columns": int(n_tgt),
        "applied": int(applied),
        "skipped_single": int(skipped_single),
    }
    return out, stats


def load_pairwise_weights(path: Path | str) -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    names = data.get("feature_names", list(FEATURE_NAMES))
    if list(names) != list(FEATURE_NAMES):
        raise ValueError(f"unexpected feature_names: {names}")
    w = data["w"]
    if len(w) != len(FEATURE_NAMES):
        raise ValueError("weights length mismatch")
    return {
        "w": [float(x) for x in w],
        "dens_min": float(data["dens_min"]),
        "gap_max": float(data["gap_max"]),
        "radius_um": float(data.get("radius_um", 15.0)),
        "voxel_scale_um": [float(x) for x in data.get("voxel_scale_um", [1.625, 0.40625, 0.40625])],
    }
