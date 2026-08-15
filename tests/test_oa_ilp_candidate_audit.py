from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from biohub_pipeline.oa_ilp_candidate_audit import (
    audit_errors,
    bucket_error,
    census_source_top1_below_gate,
    inspect_assignment,
    load_capture_frame,
)


def _error_row(**kwargs: object) -> dict[str, object]:
    base: dict[str, object] = {
        "dataset": "demo",
        "t": 0,
        "gt_source_id": 1,
        "gt_target_id": 2,
        "pred_source_id": 10,
        "pred_target_id": 20,
        "source_detected": True,
        "target_detected": True,
        "candidate_recorded": True,
        "candidate_present": False,
        "ilp_selected": False,
        "final_pair_present": False,
        "final_correct": False,
        "cause": "scorer_ranking",
        "target_rank": 2,
        "source_rank": 1,
        "blended_prob": 0.31,
    }
    base.update(kwargs)
    return base


def test_bucket_splits_ranking_by_gate() -> None:
    assert bucket_error(_error_row()) == "ranking_below_gate"
    assert (
        bucket_error(_error_row(candidate_present=True, cause="scorer_ranking"))
        == "ranking_gated"
    )
    assert (
        bucket_error(
            _error_row(
                cause="candidate_threshold",
                target_rank=1,
                candidate_present=False,
            )
        )
        == "candidate_threshold"
    )
    assert (
        bucket_error(
            _error_row(
                cause="ilp_global",
                target_rank=1,
                candidate_present=True,
                source_rank=2,
            )
        )
        == "ilp_global"
    )
    assert bucket_error(_error_row(cause="detection_miss", candidate_recorded=False)) == "detection_miss"
    assert (
        bucket_error(_error_row(cause="scorer_ranking", candidate_recorded=False, target_rank=17))
        == "candidate_absent"
    )


def test_inspect_assignment_flags_source_taken(tmp_path: Path) -> None:
    path = tmp_path / "t000_to_t001.npz"
    np.savez(
        path,
        source_id=np.array([10, 10, 11], dtype=np.int64),
        target_id=np.array([20, 21, 20], dtype=np.int64),
        blended_prob=np.array([0.62, 0.71, 0.41], dtype=np.float64),
        source_rank=np.array([2, 1, 3], dtype=np.int64),
        target_rank=np.array([1, 1, 2], dtype=np.int64),
        above_threshold=np.array([True, True, True]),
    )
    frame = load_capture_frame(path)
    result = inspect_assignment(frame, {(10, 21)}, 10, 20)
    assert result["conflict"] == "source_taken"
    assert result["source_has_wrong_gated_child"] is True
    assert result["best_gated_outgoing"]["target_id"] == 21
    assert result["gated_out_from_source"] == 2


def test_audit_recommends_source_top1_admit(tmp_path: Path) -> None:
    cap = tmp_path / "capture" / "demo"
    cap.mkdir(parents=True)
    np.savez(
        cap / "nodes.npz",
        node_id=np.array([10, 20, 21], dtype=np.int64),
        t=np.array([0, 1, 1], dtype=np.int64),
        z=np.zeros(3),
        y=np.zeros(3),
        x=np.zeros(3),
    )
    np.savez(
        cap / "t000_to_t001.npz",
        source_id=np.array([10, 11], dtype=np.int64),
        target_id=np.array([20, 20], dtype=np.int64),
        blended_prob=np.array([0.32, 0.55], dtype=np.float64),
        source_rank=np.array([1, 2], dtype=np.int64),
        target_rank=np.array([2, 1], dtype=np.int64),
        above_threshold=np.array([False, True]),
    )
    diagnostic = pd.DataFrame(
        [
            _error_row(t=0, pred_source_id=10, pred_target_id=20, source_rank=1)
            for _ in range(6)
        ]
        + [
            _error_row(
                t=0,
                cause="ilp_global",
                target_rank=1,
                source_rank=2,
                candidate_present=True,
                blended_prob=0.6,
                pred_source_id=10,
                pred_target_id=21,
            )
        ]
    )
    diagnostic["cause"] = diagnostic["cause"]  # keep columns stable
    table, summary = audit_errors(
        diagnostic,
        cap,
        dataset="demo",
        selected={(11, 20)},
    )
    assert int((table["bucket"] == "ranking_below_gate").sum()) == 6
    assert summary["ranking_below_gate_source_top1"] == 6
    assert summary["recommended_next"]["experiment"] == "admit_source_top1_below_gate"
    assert "edge-scorer retrain" in summary["recommended_next"]["do_not"]


def test_census_marks_source_top1_admit_unsafe() -> None:
    frames = {
        0: pd.DataFrame(
            {
                "source_id": np.arange(60, dtype=np.int64),
                "target_id": np.arange(100, 160, dtype=np.int64),
                "blended_prob": np.full(60, 0.2),
                "source_rank": np.ones(60, dtype=np.int64),
                "target_rank": np.ones(60, dtype=np.int64),
                "above_threshold": np.zeros(60, dtype=bool),
            }
        )
    }
    census = census_source_top1_below_gate(frames, {(0, 100)}, set())
    assert census["source_top1_below_gate"] == 60
    assert census["admit_source_top1_unsafe"] is True
