#!/usr/bin/env python3
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import numpy as np

from biohub_pipeline.config import load_config
from biohub_pipeline.inference import build_predict_command
from biohub_pipeline.margin_gated_distance_rank import apply_margin_gated_distance_rank


def main() -> None:
    cfg = load_config("configs/experiments/recipe_c_edge_0_40_margin_gated_dist_v1.yaml")
    cfg0 = load_config("configs/experiments/recipe_c_motion_off_edge_0_40_det0_96875.yaml")
    assert cfg.postprocessing["output_motion_relink"] is False
    assert abs(float(cfg.inference["edge_threshold"]) - 0.4) < 1e-12
    assert float(cfg.inference["margin_gated_dist_lambda"]) == 0.1
    real_w = Path("data/support") / cfg0.inference["weights_relative"]

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        repo = td_path / "repo"
        shutil.copytree("data/support/repo", repo)
        cmd, _ = build_predict_command(cfg, td_path / "data", repo, real_w, ["x"])
        assert "--margin-gated-dist-lambda" in cmd
        assert cmd[cmd.index("--margin-gated-dist-lambda") + 1] == "0.1"
        src = (repo / str(cfg.inference["prediction_script"])).read_text(encoding="utf-8")
        assert "_V106_MARGIN_GATED_DISTANCE_RANK_PATCH = True" in src
        print("CANDIDATE_OK")

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        repo = td_path / "repo"
        shutil.copytree("data/support/repo", repo)
        cmd0, _ = build_predict_command(cfg0, td_path / "data", repo, real_w, ["x"])
        assert "--margin-gated-dist-lambda" not in cmd0
        src = (repo / str(cfg0.inference["prediction_script"])).read_text(encoding="utf-8")
        assert "_V106_MARGIN_GATED_DISTANCE_RANK_PATCH" not in src
        print("BASELINE_OK")

    src_xy = np.array([[0.0, 0.0, 0.0], [0.0, 10.0, 0.0]])
    tgt_xy = np.array([[0.0, 0.0, 0.0]])
    logits = np.array([[1.0], [0.95]])
    out, stats = apply_margin_gated_distance_rank(
        logits,
        src_xy,
        tgt_xy,
        voxel_zyx=(1.0, 1.0, 1.0),
        lam=0.1,
        delta=0.15,
        dens_min=1.0,
    )
    assert out[1, 0] > out[0, 0]
    assert stats["applied"] == 1
    print("UNIT_OK")


if __name__ == "__main__":
    main()
