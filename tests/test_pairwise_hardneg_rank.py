from __future__ import annotations

import numpy as np

from biohub_pipeline.pairwise_hardneg_rank import apply_pairwise_hardneg_rank


def _base_case():
    # Near-tie: close distractor slightly ahead; far GT with lower seed_min
    src = np.array([[0.0, 0.0, 0.0], [0.0, 20.0, 0.0]])  # close, far
    tgt = np.array([[0.0, 0.0, 0.0]])
    # Make dense neighborhood with many identical tgt clones... use dens_min=1 for unit test
    blended = np.array([[1.0], [0.9]])
    seed1 = np.array([[1.2], [0.4]])
    seed2 = np.array([[1.1], [0.3]])
    # weights: boost dist_n, penalize seed_min (helps far/low-confidence GT)
    w = [0.0, -0.3, 0.5, 0.0, 0.0, 0.0]
    return blended, seed1, seed2, src, tgt, w


def test_near_tie_dense_can_flip_to_farther() -> None:
    blended, seed1, seed2, src, tgt, w = _base_case()
    out, stats = apply_pairwise_hardneg_rank(
        blended,
        seed1,
        seed2,
        src,
        tgt,
        voxel_zyx=(1.0, 1.0, 1.0),
        weights=w,
        dens_min=1.0,
        gap_max=1.5,
    )
    assert stats["applied"] == 1
    assert out[1, 0] > out[0, 0]


def test_clear_winner_unchanged() -> None:
    blended, seed1, seed2, src, tgt, w = _base_case()
    blended = np.array([[3.0], [0.5]])
    out, stats = apply_pairwise_hardneg_rank(
        blended,
        seed1,
        seed2,
        src,
        tgt,
        voxel_zyx=(1.0, 1.0, 1.0),
        weights=w,
        dens_min=1.0,
        gap_max=1.5,
    )
    assert stats["applied"] == 0
    assert np.allclose(out, blended)


def test_density_gate_blocks_sparse() -> None:
    blended, seed1, seed2, src, tgt, w = _base_case()
    out, stats = apply_pairwise_hardneg_rank(
        blended,
        seed1,
        seed2,
        src,
        tgt,
        voxel_zyx=(1.0, 1.0, 1.0),
        weights=w,
        dens_min=8.0,
        gap_max=1.5,
    )
    assert stats["applied"] == 0
    assert np.allclose(out, blended)
