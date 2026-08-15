"""Mine pairwise hard-negatives from frozen-recipe candidate captures.

Training uses predicted detection IDs from the capture, not GT graph IDs.
Holdout-8 is never mined for training.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

FIXED8_DATASETS = (
    "44b6_0113de3b",
    "44b6_0b24845f",
    "44b6_341df25f",
    "44b6_e57ff5c6",
    "6bba_05b6850b",
    "6bba_05db0fb1",
    "6bba_969618f6",
    "6bba_fc83837d",
)
HOLDOUT8_DATASETS = (
    "44b6_0c582fdc",
    "44b6_0db75fae",
    "44b6_12dfb391",
    "44b6_144b256d",
    "6bba_062c8d37",
    "6bba_07477033",
    "6bba_07e24132",
    "6bba_085bf656",
)

RANK2_WEIGHT = 3.0
RANKING_WEIGHT = 2.0
CONTROL_WEIGHT = 1.0
CONTROL_GAP_MAX = 2.0
CONTROL_MIN_DENSITY = 6.0
MAX_NEGS_RANKING = 3
BASELINE_FIXED8 = 0.9181439782806684
BASELINE_HOLDOUT8 = 0.9646726188580379
MATERIAL_FIXED_GAIN = 0.001
HOLDOUT_NOISE = 0.0003


def load_capture_nodes(cap_ds: Path) -> pd.DataFrame:
    with np.load(cap_ds / "nodes.npz") as data:
        return pd.DataFrame(
            {
                "node_id": data["node_id"].astype(np.int64),
                "t": data["t"].astype(np.int64),
                "z": data["z"].astype(np.float64),
                "y": data["y"].astype(np.float64),
                "x": data["x"].astype(np.float64),
            }
        )


def _npz_by_t(cap_ds: Path) -> dict[int, Path]:
    out: dict[int, Path] = {}
    for path in cap_ds.glob("t*_to_t*.npz"):
        out[int(path.name.split("_")[0][1:])] = path
    return out


def _pair_rows_for_target(
    path: Path,
    pred_target_id: int,
    pred_gt_source_id: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    with np.load(path) as data:
        sources = data["source_id"].astype(np.int64)
        targets = data["target_id"].astype(np.int64)
        blend = data["blended_logit"].astype(np.float64)
    mask = targets == int(pred_target_id)
    if not np.any(mask):
        return None
    src = sources[mask]
    scores = blend[mask]
    gt_pos = np.flatnonzero(src == int(pred_gt_source_id))
    if len(gt_pos) == 0:
        return None
    return src, scores, gt_pos


def mine_pairs_from_capture(
    diagnostic: pd.DataFrame,
    capture_root: Path,
    *,
    datasets: tuple[str, ...] | None = None,
    max_negs_ranking: int = MAX_NEGS_RANKING,
    control_gap_max: float = CONTROL_GAP_MAX,
    control_min_density: float = CONTROL_MIN_DENSITY,
) -> pd.DataFrame:
    """Return one row per (GT source, hard-neg source, target) training pair."""
    allowed = set(datasets if datasets is not None else FIXED8_DATASETS)
    holdout = set(HOLDOUT8_DATASETS)
    rows: list[dict[str, Any]] = []
    work = diagnostic[
        diagnostic["source_detected"].astype(bool)
        & diagnostic["target_detected"].astype(bool)
        & diagnostic["dataset"].isin(allowed)
    ].copy()
    for dataset, group in work.groupby("dataset", sort=False):
        if str(dataset) in holdout:
            continue
        cap_ds = capture_root / str(dataset)
        if not cap_ds.exists():
            continue
        npz_map = _npz_by_t(cap_ds)
        for rec in group.itertuples(index=False):
            if pd.isna(rec.pred_source_id) or pd.isna(rec.pred_target_id):
                continue
            fpath = npz_map.get(int(rec.t))
            if fpath is None:
                continue
            parsed = _pair_rows_for_target(
                fpath, int(rec.pred_target_id), int(rec.pred_source_id)
            )
            if parsed is None:
                continue
            src, scores, gt_pos = parsed
            gi = int(gt_pos[0])
            other = [i for i in range(len(src)) if i != gi]
            if not other:
                continue
            cause = str(rec.cause)
            rank = int(rec.target_rank) if pd.notna(rec.target_rank) else -1
            dens = float(rec.local_density_pred) if pd.notna(rec.local_density_pred) else 0.0
            ranked = sorted(other, key=lambda i: float(scores[i]), reverse=True)
            order = np.argsort(-scores)
            gap01 = (
                float(scores[order[0]] - scores[order[1]]) if len(scores) > 1 else 0.0
            )
            if cause == "scorer_ranking":
                weight = RANK2_WEIGHT if rank == 2 else RANKING_WEIGHT
                negs = ranked[:max_negs_ranking]
            elif cause == "correct":
                if gap01 > control_gap_max and dens < control_min_density:
                    continue
                weight = CONTROL_WEIGHT
                negs = ranked[:1]
            else:
                continue
            for ni in negs:
                rows.append(
                    {
                        "dataset": str(dataset),
                        "t": int(rec.t),
                        "pred_target_id": int(rec.pred_target_id),
                        "pred_gt_source_id": int(rec.pred_source_id),
                        "pred_neg_source_id": int(src[ni]),
                        "cause": cause,
                        "cohort": "rank2"
                        if cause == "scorer_ranking" and rank == 2
                        else ("ranking" if cause == "scorer_ranking" else "control"),
                        "weight": float(weight),
                        "target_rank": rank,
                        "density": dens,
                        "gap01": gap01,
                        "gt_logit": float(scores[gi]),
                        "neg_logit": float(scores[ni]),
                    }
                )
    columns = [
        "dataset",
        "t",
        "pred_target_id",
        "pred_gt_source_id",
        "pred_neg_source_id",
        "cause",
        "cohort",
        "weight",
        "target_rank",
        "density",
        "gap01",
        "gt_logit",
        "neg_logit",
    ]
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns)


def summarize_pairs(pairs: pd.DataFrame) -> dict[str, Any]:
    if pairs.empty:
        return {"n_pairs": 0, "n_windows": 0, "by_cohort": {}, "by_dataset": {}}
    return {
        "n_pairs": len(pairs),
        "n_windows": int(pairs.groupby(["dataset", "t"]).ngroups),
        "n_rank2_pairs": int((pairs["cohort"] == "rank2").sum()),
        "n_ranking_pairs": int(pairs["cohort"].isin(["rank2", "ranking"]).sum()),
        "n_control_pairs": int((pairs["cohort"] == "control").sum()),
        "by_cohort": {str(k): int(v) for k, v in pairs["cohort"].value_counts().items()},
        "by_dataset": {str(k): int(v) for k, v in pairs["dataset"].value_counts().items()},
        "datasets": sorted(pairs["dataset"].unique().tolist()),
    }


def decide_promotion(
    fixed_score: float,
    holdout_score: float,
    *,
    fixed_baseline: float = BASELINE_FIXED8,
    holdout_baseline: float = BASELINE_HOLDOUT8,
) -> dict[str, Any]:
    """Promote only if both beat baseline, or a clear fixed gain with no material holdout drop."""
    fixed_delta = float(fixed_score) - float(fixed_baseline)
    holdout_delta = float(holdout_score) - float(holdout_baseline)
    both_beat = fixed_delta > 0.0 and holdout_delta > 0.0
    material_fixed_no_holdout_drop = (
        fixed_delta >= MATERIAL_FIXED_GAIN and holdout_delta >= -HOLDOUT_NOISE
    )
    promote = bool(both_beat or material_fixed_no_holdout_drop)
    if promote:
        decision = "PROMOTE"
        next_step = (
            "Replace production edge checkpoints with hardneg_v1 under the frozen recipe "
            "(motion OFF, edge_threshold=0.40). Do not change gates."
        )
    else:
        decision = "REJECT"
        next_step = (
            "Do not start another edge-scorer retrain. Inspect remaining ordinary-association "
            "failures on holdout 44b6_12dfb391 with frozen detections (ILP / candidate set), "
            "not another ranking-weight or architecture sweep."
        )
    return {
        "decision": decision,
        "promote": promote,
        "fixed_score": float(fixed_score),
        "holdout_score": float(holdout_score),
        "fixed_baseline": float(fixed_baseline),
        "holdout_baseline": float(holdout_baseline),
        "fixed_delta": fixed_delta,
        "holdout_delta": holdout_delta,
        "next_step": next_step,
    }
