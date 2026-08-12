"""Margin-gated distance compensation for softmax edge ranking.

Softmax association scores systematically prefer nearby distractors. Rank-2
ordinary-association failures are typically much farther than the winning
competitor. A global +λ·dist term destroys short true edges; this module only
boosts distances inside ambiguous (near-tie) columns in dense neighborhoods.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


def pairwise_dist_um(
    src_zyx: np.ndarray,
    tgt_zyx: np.ndarray,
    voxel_zyx: Sequence[float],
) -> np.ndarray:
    """Return (n_src, n_tgt) Euclidean distances in µm."""
    scale = np.asarray(voxel_zyx, dtype=np.float64).reshape(1, 1, 3)
    src = np.asarray(src_zyx, dtype=np.float64).reshape(-1, 1, 3)
    tgt = np.asarray(tgt_zyx, dtype=np.float64).reshape(1, -1, 3)
    return np.linalg.norm((src - tgt) * scale, axis=2)


def target_densities_um(
    tgt_zyx: np.ndarray,
    voxel_zyx: Sequence[float],
    radius_um: float,
) -> np.ndarray:
    """Count target-frame detections within radius_um of each target (incl. self)."""
    scale = np.asarray(voxel_zyx, dtype=np.float64).reshape(1, 1, 3)
    pts = np.asarray(tgt_zyx, dtype=np.float64).reshape(-1, 3)
    if len(pts) == 0:
        return np.zeros(0, dtype=np.int32)
    d = np.linalg.norm((pts[:, None, :] - pts[None, :, :]) * scale, axis=2)
    return (d <= float(radius_um)).sum(axis=1).astype(np.int32)


def apply_margin_gated_distance_rank(
    logits: np.ndarray,
    src_zyx: np.ndarray,
    tgt_zyx: np.ndarray,
    *,
    voxel_zyx: Sequence[float],
    lam: float,
    delta: float,
    dens_min: float,
    radius_um: float = 15.0,
) -> tuple[np.ndarray, dict[str, int]]:
    """Return adjusted logits and simple apply counters.

    For each target column, if (top1_logit - top2_logit) < delta and local
    density >= dens_min, add ``lam * dist_um`` to every source logit in that
    column. Clear winners are unchanged.
    """
    raw = np.asarray(logits, dtype=np.float64)
    if raw.ndim != 2:
        raise ValueError(f"logits must be 2D, got shape {raw.shape}")
    n_src, n_tgt = raw.shape
    if n_src == 0 or n_tgt == 0:
        return raw.copy(), {"columns": 0, "applied": 0, "skipped_single": 0}

    dists = pairwise_dist_um(src_zyx, tgt_zyx, voxel_zyx)
    dens = target_densities_um(tgt_zyx, voxel_zyx, radius_um)
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
        if gap < float(delta) and float(dens[j]) >= float(dens_min):
            out[:, j] = col + float(lam) * dists[:, j]
            applied += 1
    stats = {
        "columns": int(n_tgt),
        "applied": int(applied),
        "skipped_single": int(skipped_single),
    }
    return out, stats
