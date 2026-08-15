#!/usr/bin/env python3
"""Compare hard-neg retrain eval vs frozen baseline capture; write decision report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from biohub_pipeline.edge_hardneg_mine import (
    BASELINE_FIXED8,
    BASELINE_HOLDOUT8,
    decide_promotion,
)


def _score(path: Path) -> dict:
    return json.loads(path.read_text())


def _metric_row(path: Path, dataset: str) -> dict | None:
    frame = pd.read_csv(path)
    part = frame[frame["dataset"] == dataset]
    if part.empty:
        return None
    row = part.iloc[0]
    return {
        "dataset": dataset,
        "adj_edge_jaccard": float(row["adj_edge_jaccard"]),
        "edge_tp": int(row["edge_tp"]),
        "edge_fp": int(row["edge_fp"]),
        "edge_fn": int(row["edge_fn"]),
    }


def _ranking_stats(diag: pd.DataFrame) -> dict:
    em = diag[diag["source_detected"].astype(bool) & diag["target_detected"].astype(bool)]
    causal = em[em["cause"] != "correct"]
    ranking = em[em["cause"] == "scorer_ranking"]
    rank2 = ranking[ranking["target_rank"] == 2]
    return {
        "ordinary": len(diag),
        "causal_oa": len(causal),
        "scorer_ranking": len(ranking),
        "rank2": len(rank2),
        "ranking_share_causal": float(len(ranking) / len(causal)) if len(causal) else None,
    }


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-root", type=Path, required=True)
    p.add_argument("--baseline-fixed-diag", type=Path, required=True)
    p.add_argument("--baseline-hold-diag", type=Path, required=True)
    p.add_argument("--baseline-fixed-metrics", type=Path, required=True)
    p.add_argument("--baseline-hold-metrics", type=Path, required=True)
    p.add_argument("--job-id", default="")
    p.add_argument("--git-commit", default="unknown")
    return p


def main() -> None:
    args = _parser().parse_args()
    root = args.run_root
    fixed = _score(root / "fixed8" / "score_summary.json")
    hold = _score(root / "holdout8" / "score_summary.json")
    decision = decide_promotion(float(fixed["score"]), float(hold["score"]))
    new_fixed_diag = pd.read_csv(root / "fixed8" / "candidate_edge_diagnostic.csv")
    new_hold_diag = pd.read_csv(root / "holdout8" / "candidate_edge_diagnostic.csv")
    base_fixed_diag = pd.read_csv(args.baseline_fixed_diag)
    base_hold_diag = pd.read_csv(args.baseline_hold_diag)
    key_ds = "44b6_12dfb391"
    comparison = {
        "experiment": "edge_scorer_hardneg_retrain_v1",
        "frozen_recipe": "recipe_c_motion_off_edge_0_40_det0_96875",
        "job_id": args.job_id,
        "git_commit": args.git_commit,
        **decision,
        "fixed8": {
            "score": float(fixed["score"]),
            "edge_tp_fp_fn": [int(fixed["edge_tp"]), int(fixed["edge_fp"]), int(fixed["edge_fn"])],
            "baseline_ranking": _ranking_stats(base_fixed_diag),
            "new_ranking": _ranking_stats(new_fixed_diag),
        },
        "holdout8": {
            "score": float(hold["score"]),
            "edge_tp_fp_fn": [int(hold["edge_tp"]), int(hold["edge_fp"]), int(hold["edge_fn"])],
            "baseline_ranking": _ranking_stats(base_hold_diag),
            "new_ranking": _ranking_stats(new_hold_diag),
            "dataset_44b6_12dfb391": {
                "baseline": _metric_row(args.baseline_hold_metrics, key_ds),
                "new": _metric_row(root / "holdout8" / "metric_by_dataset.csv", key_ds),
            },
        },
        "baseline_fixed8": BASELINE_FIXED8,
        "baseline_holdout8": BASELINE_HOLDOUT8,
    }
    (root / "comparison.json").write_text(json.dumps(comparison, indent=2, sort_keys=True) + "\n")

    f_rank = comparison["fixed8"]["new_ranking"]
    f_base = comparison["fixed8"]["baseline_ranking"]
    h_rank = comparison["holdout8"]["new_ranking"]
    h_base = comparison["holdout8"]["baseline_ranking"]
    d12_b = comparison["holdout8"]["dataset_44b6_12dfb391"]["baseline"] or {}
    d12_n = comparison["holdout8"]["dataset_44b6_12dfb391"]["new"] or {}
    report = f"""# Edge-scorer hard-neg retrain v1 — {decision['decision']}

**Experiment:** `edge_scorer_hardneg_retrain_v1`
**Frozen recipe:** motion-relink OFF, `edge_threshold=0.40` (unchanged)
**Job:** slurm `{args.job_id}`  **Commit:** `{args.git_commit}`

## Decision: **{decision['decision']}**

{decision['next_step']}

## What changed

Fine-tuned both two-seed **edge transformers** with pairwise ranking loss on
rank-2 near-misses + controls mined from the frozen-recipe capture
(`candidate_capture_recipe_c_edge_0_40_v1`, fixed-8 only). UNet / detect_head
frozen. Holdout-8 was not used for training. Linear pairwise reweight was not
re-enabled.

Eval config: `configs/experiments/recipe_c_edge_0_40_hardneg_retrain_v1.yaml`

## Scores

| Split | Frozen baseline | Hard-neg retrain | Δ |
| --- | ---: | ---: | ---: |
| fixed-8 | {BASELINE_FIXED8:.16f} | {decision['fixed_score']:.16f} | {decision['fixed_delta']:+.6f} |
| holdout-8 | {BASELINE_HOLDOUT8:.16f} | {decision['holdout_score']:.16f} | {decision['holdout_delta']:+.6f} |

Edges TP/FP/FN: fixed {comparison['fixed8']['edge_tp_fp_fn']} ; holdout {comparison['holdout8']['edge_tp_fp_fn']}

## Causal ranking

| Split | ranking (base → new) | rank-2 (base → new) | causal OA ranking share |
| --- | --- | --- | --- |
| fixed-8 | {f_base['scorer_ranking']} → {f_rank['scorer_ranking']} | {f_base['rank2']} → {f_rank['rank2']} | {f_base['ranking_share_causal']} → {f_rank['ranking_share_causal']} |
| holdout-8 | {h_base['scorer_ranking']} → {h_rank['scorer_ranking']} | {h_base['rank2']} → {h_rank['rank2']} | {h_base['ranking_share_causal']} → {h_rank['ranking_share_causal']} |

### `44b6_12dfb391`

- baseline adj={d12_b.get('adj_edge_jaccard')} TP/FP/FN={d12_b.get('edge_tp')}/{d12_b.get('edge_fp')}/{d12_b.get('edge_fn')}
- new adj={d12_n.get('adj_edge_jaccard')} TP/FP/FN={d12_n.get('edge_tp')}/{d12_n.get('edge_fp')}/{d12_n.get('edge_fn')}

## Artifacts

- NFS: `~/biohub-outputs/experiments/edge_scorer_hardneg_retrain_v1/`
- Login compact: `outputs/experiments/edge_scorer_hardneg_retrain_v1/`
- Report: `outputs/analysis/edge_scorer_hardneg_retrain_v1_report.md`
"""
    (root / "report.md").write_text(report)
    analysis = Path("outputs/analysis")
    analysis.mkdir(parents=True, exist_ok=True)
    (analysis / "edge_scorer_hardneg_retrain_v1_report.md").write_text(report)
    print(json.dumps(comparison, indent=2, sort_keys=True))
    print("DECISION", decision["decision"], flush=True)


if __name__ == "__main__":
    main()
