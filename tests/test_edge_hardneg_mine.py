from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from biohub_pipeline.config import load_config
from biohub_pipeline.edge_hardneg_mine import (
    decide_promotion,
    mine_pairs_from_capture,
    summarize_pairs,
)


def _write_capture(root: Path, dataset: str) -> None:
    ds = root / dataset
    ds.mkdir(parents=True)
    np.savez(
        ds / "nodes.npz",
        node_id=np.array([10, 11, 12, 20], dtype=np.int64),
        t=np.array([0, 0, 0, 1], dtype=np.int64),
        z=np.array([0, 0, 0, 0], dtype=np.float64),
        y=np.array([0, 4, 8, 0], dtype=np.float64),
        x=np.array([0, 0, 0, 0], dtype=np.float64),
    )
    # Target 20: GT source 10 (logit 1.0, rank 2), competitor 11 (1.2), third 12 (0.4)
    np.savez(
        ds / "t000_to_t001.npz",
        source_id=np.array([10, 11, 12], dtype=np.int64),
        target_id=np.array([20, 20, 20], dtype=np.int64),
        blended_logit=np.array([1.0, 1.2, 0.4], dtype=np.float64),
        target_rank=np.array([2, 1, 3], dtype=np.int64),
    )


def _row(**kwargs: object) -> dict[str, object]:
    base: dict[str, object] = {
        "dataset": "44b6_0113de3b",
        "t": 0,
        "pred_source_id": 10,
        "pred_target_id": 20,
        "source_detected": True,
        "target_detected": True,
        "cause": "scorer_ranking",
        "target_rank": 2,
        "local_density_pred": 12.0,
    }
    base.update(kwargs)
    return base


def test_rank2_near_miss_emits_weighted_pairs(tmp_path: Path) -> None:
    _write_capture(tmp_path, "44b6_0113de3b")
    diag = pd.DataFrame([_row()])
    pairs = mine_pairs_from_capture(diag, tmp_path)
    assert len(pairs) == 2
    assert set(pairs["pred_neg_source_id"].astype(int)) == {11, 12}
    assert (pairs["cohort"] == "rank2").all()
    assert (pairs["weight"] == 3.0).all()


def test_controls_keep_near_ties_skip_easy(tmp_path: Path) -> None:
    _write_capture(tmp_path, "44b6_0113de3b")
    near = _row(cause="correct", target_rank=1, local_density_pred=9.0)
    # Easy sparse success with a large logit gap must not enter the control set.
    _write_capture(tmp_path, "6bba_05b6850b")
    ds = tmp_path / "6bba_05b6850b"
    np.savez(
        ds / "t000_to_t001.npz",
        source_id=np.array([10, 11], dtype=np.int64),
        target_id=np.array([20, 20], dtype=np.int64),
        blended_logit=np.array([3.0, 0.1], dtype=np.float64),
        target_rank=np.array([1, 2], dtype=np.int64),
    )
    easy = _row(
        dataset="6bba_05b6850b",
        cause="correct",
        target_rank=1,
        local_density_pred=2.0,
        pred_source_id=10,
    )
    pairs = mine_pairs_from_capture(pd.DataFrame([near, easy]), tmp_path)
    assert (pairs["dataset"] == "44b6_0113de3b").all()
    assert (pairs["cohort"] == "control").all()
    assert len(pairs) == 1
    assert int(pairs.iloc[0]["pred_neg_source_id"]) == 11


def test_holdout_datasets_are_never_mined(tmp_path: Path) -> None:
    _write_capture(tmp_path, "44b6_12dfb391")
    diag = pd.DataFrame([_row(dataset="44b6_12dfb391")])
    pairs = mine_pairs_from_capture(diag, tmp_path)
    assert pairs.empty


def test_hardneg_eval_config_keeps_frozen_gates() -> None:
    frozen = load_config("configs/experiments/recipe_c_motion_off_edge_0_40_det0_96875.yaml")
    cand = load_config("configs/experiments/recipe_c_edge_0_40_hardneg_retrain_v1.yaml")
    assert frozen.postprocessing["output_motion_relink"] is False
    assert cand.postprocessing["output_motion_relink"] is False
    assert abs(float(frozen.inference["edge_threshold"]) - 0.4) < 1e-12
    assert abs(float(cand.inference["edge_threshold"]) - 0.4) < 1e-12
    assert cand.inference.get("pairwise_hardneg_w") is None
    assert "hardneg_v1_split_0" in cand.inference["weights_relative"]
    assert "hardneg_v1_seed_314159" in cand.inference["ensemble_weights_relative"]
    assert frozen.inference["weights_relative"].endswith("split_0/edge_predictor_best.pth")


def test_summarize_and_promotion_rules() -> None:
    pairs = pd.DataFrame(
        {
            "dataset": ["44b6_0113de3b", "6bba_05db0fb1"],
            "t": [0, 1],
            "cohort": ["rank2", "control"],
        }
    )
    summary = summarize_pairs(pairs)
    assert summary["n_pairs"] == 2
    assert summary["n_rank2_pairs"] == 1
    promote = decide_promotion(0.920, 0.965)
    assert promote["decision"] == "PROMOTE"
    reject = decide_promotion(0.924, 0.9638)
    assert reject["decision"] == "REJECT"
    assert "Do not start another" in reject["next_step"]
    # Material fixed gain with holdout inside noise band.
    noise = decide_promotion(0.920, 0.9646726188580379 - 0.0001)
    assert noise["decision"] == "PROMOTE"
