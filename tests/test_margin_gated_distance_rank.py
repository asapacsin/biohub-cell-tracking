from __future__ import annotations

import numpy as np

from biohub_pipeline.margin_gated_distance_rank import apply_margin_gated_distance_rank


def test_near_tie_prefers_farther_after_bonus() -> None:
    src = np.array([[0.0, 0.0, 0.0], [0.0, 10.0, 0.0]])
    tgt = np.array([[0.0, 0.0, 0.0]])
    logits = np.array([[1.0], [0.95]])
    out, stats = apply_margin_gated_distance_rank(
        logits,
        src,
        tgt,
        voxel_zyx=(1.0, 1.0, 1.0),
        lam=0.1,
        delta=0.15,
        dens_min=1.0,
        radius_um=15.0,
    )
    assert stats["applied"] == 1
    assert out[1, 0] > out[0, 0]


def test_clear_winner_unchanged() -> None:
    src = np.array([[0.0, 0.0, 0.0], [0.0, 10.0, 0.0]])
    tgt = np.array([[0.0, 0.0, 0.0]])
    logits = np.array([[3.0], [0.5]])
    out, stats = apply_margin_gated_distance_rank(
        logits,
        src,
        tgt,
        voxel_zyx=(1.0, 1.0, 1.0),
        lam=0.1,
        delta=0.15,
        dens_min=1.0,
    )
    assert stats["applied"] == 0
    assert np.allclose(out, logits)


def test_density_gate_blocks_sparse() -> None:
    src = np.array([[0.0, 0.0, 0.0], [0.0, 10.0, 0.0]])
    tgt = np.array([[0.0, 0.0, 0.0]])
    logits = np.array([[1.0], [0.95]])
    out, stats = apply_margin_gated_distance_rank(
        logits,
        src,
        tgt,
        voxel_zyx=(1.0, 1.0, 1.0),
        lam=0.1,
        delta=0.15,
        dens_min=8.0,
        radius_um=15.0,
    )
    assert stats["applied"] == 0
    assert np.allclose(out, logits)
