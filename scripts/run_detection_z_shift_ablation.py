#!/usr/bin/env python3
"""Sweep integer z-voxel shifts on frozen-recipe postprocessed submissions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from biohub_pipeline.detection_z_shift import evaluate_prediction_frame, shift_node_z
from biohub_pipeline.fixed8_cv import FIXED8_DATASETS

HOLDOUT8 = [
    "44b6_0c582fdc",
    "44b6_0db75fae",
    "44b6_12dfb391",
    "44b6_144b256d",
    "6bba_062c8d37",
    "6bba_07477033",
    "6bba_07e24132",
    "6bba_085bf656",
]

BASELINE = {"fixed8": 0.9181439782806684, "holdout8": 0.9646726188580379}


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--fixed8-csv",
        type=Path,
        default=Path(
            "outputs/experiments/candidate_capture_recipe_c_edge_0_40_v1/fixed8/predictions/postprocessed_submission.csv"
        ),
    )
    p.add_argument(
        "--holdout8-csv",
        type=Path,
        default=Path(
            "outputs/experiments/candidate_capture_recipe_c_edge_0_40_v1/holdout8/predictions/postprocessed_submission.csv"
        ),
    )
    p.add_argument("--data-dir", type=Path, default=Path("data/competition/train"))
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/experiments/detection_z_shift_v1"),
    )
    p.add_argument("--shifts", type=int, nargs="*", default=[-2, -1, 0, 1, 2, 3, 4])
    return p


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, float) and (pd.isna(value) or value in (float("inf"), float("-inf"))):
        return None
    if hasattr(value, "item"):
        return _json_safe(value.item())
    return value


def main() -> None:
    args = _parser().parse_args()
    data_dir = args.data_dir
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    splits = {
        "fixed8": (
            pd.read_csv(args.fixed8_csv),
            list(FIXED8_DATASETS),
            BASELINE["fixed8"],
            Path("outputs/experiments/candidate_capture_recipe_c_edge_0_40_v1/fixed8/metric_by_dataset.csv"),
        ),
        "holdout8": (
            pd.read_csv(args.holdout8_csv),
            list(HOLDOUT8),
            BASELINE["holdout8"],
            Path("outputs/experiments/candidate_capture_recipe_c_edge_0_40_v1/holdout8/metric_by_dataset.csv"),
        ),
    }
    rows: list[dict[str, Any]] = []
    by_shift: dict[str, Any] = {}
    for delta in args.shifts:
        by_shift[str(delta)] = {}
        for split, (frame, datasets, baseline, metric_path) in splits.items():
            estimates = {
                str(r.dataset): float(r.estimated_total_nodes)
                for r in pd.read_csv(metric_path).itertuples(index=False)
            }
            shifted = shift_node_z(frame, delta)
            per, summary = evaluate_prediction_frame(
                shifted, data_dir, datasets, estimated_total_nodes=estimates
            )
            pd.DataFrame(per).to_csv(out / f"{split}_shift_{delta:+d}_per_dataset.csv", index=False)
            score = float(summary["score"])
            rec = {
                "split": split,
                "delta_voxels": delta,
                "score": score,
                "delta_vs_baseline": score - baseline,
                "edge_tp": int(summary["edge_tp"]),
                "edge_fp": int(summary["edge_fp"]),
                "edge_fn": int(summary["edge_fn"]),
                "node_recall": float(summary["node_recall"]),
            }
            killer = next((r for r in per if r["dataset"] == "6bba_07e24132"), None)
            if killer is not None:
                rec["dataset_6bba_07e24132"] = float(killer["adj_edge_jaccard"])
                rec["dataset_6bba_07e24132_node_recall"] = float(killer["node_recall"])
            fc = next((r for r in per if r["dataset"] == "6bba_fc83837d"), None)
            if fc is not None:
                rec["dataset_6bba_fc83837d"] = float(fc["adj_edge_jaccard"])
                rec["dataset_6bba_fc83837d_node_recall"] = float(fc["node_recall"])
            rows.append(rec)
            by_shift[str(delta)][split] = rec
            print(
                f"shift {delta:+d} {split:8s} score={score:.6f} Δ={score-baseline:+.6f} "
                f"rec={rec['node_recall']:.4f} tp/fp/fn={rec['edge_tp']}/{rec['edge_fp']}/{rec['edge_fn']}"
            )

    table = pd.DataFrame(rows)
    table.to_csv(out / "sweep.csv", index=False)

    # Promote only if some nonzero shift is >= baseline on BOTH splits.
    best = None
    for delta in args.shifts:
        if delta == 0:
            continue
        f = by_shift[str(delta)]["fixed8"]["delta_vs_baseline"]
        h = by_shift[str(delta)]["holdout8"]["delta_vs_baseline"]
        if f >= -1e-12 and h >= -1e-12:
            if best is None or (f + h) > (
                by_shift[str(best)]["fixed8"]["delta_vs_baseline"]
                + by_shift[str(best)]["holdout8"]["delta_vs_baseline"]
            ):
                best = delta
    decision = {
        "promote": best is not None,
        "best_nonnegative_both_splits": best,
        "rule": "Promote a z-shift only if fixed-8 AND holdout-8 are not below frozen baseline.",
        "baseline": BASELINE,
        "by_shift": by_shift,
    }
    (out / "decision.json").write_text(
        json.dumps(_json_safe(decision), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("DECISION", json.dumps({"promote": decision["promote"], "best": best}))


if __name__ == "__main__":
    main()
