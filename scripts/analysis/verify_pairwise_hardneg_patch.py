#!/usr/bin/env python3
"""Sanity-check pairwise hardneg module + config load."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from biohub_pipeline.config import load_config
from biohub_pipeline.pairwise_hardneg_rank import apply_pairwise_hardneg_rank, load_pairwise_weights


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    cfg = load_config(root / "configs/experiments/recipe_c_edge_0_40_pairwise_hardneg_v1.yaml")
    assert abs(float(cfg.inference["edge_threshold"]) - 0.4) < 1e-12
    assert cfg.postprocessing["output_motion_relink"] is False
    w = cfg.inference["pairwise_hardneg_w"]
    assert len(w) == 6
    weights = load_pairwise_weights(root / "outputs/analysis/pairwise_hardneg_rank_v1/pairwise_weights.json")
    assert len(weights["w"]) == 6
    # smoke apply
    src = np.array([[0.0, 0.0, 0.0], [0.0, 10.0, 0.0]])
    tgt = np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 2.0, 0.0], [0.0, 3.0, 0.0],
                    [0.0, 4.0, 0.0], [0.0, 5.0, 0.0], [0.0, 6.0, 0.0], [0.0, 7.0, 0.0]])
    n_t = len(tgt)
    blended = np.ones((2, n_t))
    blended[0] = 1.0
    blended[1] = 0.95
    seed1 = blended.copy()
    seed2 = blended.copy() * 0.9
    out, stats = apply_pairwise_hardneg_rank(
        blended, seed1, seed2, src, tgt,
        voxel_zyx=(1.0, 1.0, 1.0),
        weights=weights["w"],
        dens_min=weights["dens_min"],
        gap_max=weights["gap_max"],
        radius_um=15.0,
    )
    assert out.shape == blended.shape
    print("OK", json.dumps({"stats": stats, "dens_min": weights["dens_min"], "gap_max": weights["gap_max"]}))


if __name__ == "__main__":
    main()
