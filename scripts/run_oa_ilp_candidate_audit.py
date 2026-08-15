#!/usr/bin/env python3
"""CPU audit of remaining OA failures on one holdout dataset (frozen detections)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from biohub_pipeline.oa_ilp_candidate_audit import audit_errors

DEFAULT_DATASET = "44b6_12dfb391"
DEFAULT_CAPTURE = Path(
    "outputs/experiments/candidate_capture_recipe_c_edge_0_40_v1/holdout8"
)


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default=DEFAULT_DATASET)
    p.add_argument("--capture-root", type=Path, default=DEFAULT_CAPTURE)
    p.add_argument(
        "--geff",
        type=Path,
        default=Path("outputs/experiments/oa_ilp_candidate_audit_v1/raw_geff")
        / f"{DEFAULT_DATASET}.geff",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/experiments/oa_ilp_candidate_audit_v1"),
    )
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


def _write_report(path: Path, summary: dict[str, Any], table: pd.DataFrame) -> None:
    rec = summary["recommended_next"]
    buckets = summary["bucket_counts"]
    recover = summary["recoverability_counts"]
    lines = [
        f"# OA ILP / candidate-set audit — `{summary['dataset']}`",
        "",
        "**Frozen recipe:** motion-relink OFF, `edge_threshold=0.40`. Detections and scorer weights unchanged.",
        "**Question:** After ranking retrains failed to transfer, are remaining ordinary-association failures a candidate-set hole (GT never admitted to ILP) or an ILP occupancy conflict?",
        "",
        "## Decision",
        "",
        f"**Next experiment:** `{rec['experiment']}`",
        "",
        rec["rationale"],
        "",
        "Do not: " + "; ".join(rec["do_not"]) + ".",
        "",
        "## Dataset burden",
        "",
        f"- Ordinary associations: {summary['ordinary_associations']}",
        f"- Errors: {summary['errors']} (causal / endpoint-matched: {summary['causal_errors']})",
        f"- Selected-edge source: `{summary['selected_edge_source']}`",
        "",
        "## Causal buckets (endpoint-matched unless detection)",
        "",
        "| Bucket | Count | Meaning |",
        "| --- | ---: | --- |",
        f"| ranking_below_gate | {buckets['ranking_below_gate']} | GT recorded, rank>1, softmax ≤ 0.40 (not in ILP graph) |",
        f"| candidate_threshold | {buckets['candidate_threshold']} | GT rank-1 but softmax ≤ 0.40 |",
        f"| candidate_absent | {buckets['candidate_absent']} | GT not in top-k ∪ gated capture |",
        f"| ranking_gated | {buckets['ranking_gated']} | GT rank>1 but above 0.40 (ILP could have chosen it) |",
        f"| ilp_global | {buckets['ilp_global']} | GT rank-1 and gated; ILP selected a different occupancy |",
        f"| postprocessing_rematch | {buckets['postprocessing_rematch']} | ILP kept the pair; final identity mapping missed GT |",
        f"| detection_miss | {buckets['detection_miss']} | Frozen; not an ILP/candidate lever |",
        "",
        "## Shares of causal errors",
        "",
        f"- Never admitted to ILP (below gate / absent): **{summary['causal_share_below_gate']:.1%}**",
        f"- In ILP graph (gated ranking + ILP occupancy): **{summary['causal_share_in_ilp_graph']:.1%}**",
        f"- Pure ILP occupancy: **{summary['causal_share_ilp_contention']:.1%}**",
        f"- Rematch: **{summary['causal_share_rematch']:.1%}**",
        "",
        "## Source-top-1 below-gate census (admit-all FP risk)",
        "",
        f"- Source-top-1 pairs with softmax ≤ 0.40: **{summary['source_top1_below_gate']}**",
        f"- Of those, GT ranking/threshold errors: {summary['source_top1_below_gate_gt_err']}",
        f"- Of those, already-correct GT pairs (should not happen if gated): {summary['source_top1_below_gate_gt_ok']}",
        f"- Median softmax of excluded top-1: {summary['source_top1_below_gate_median_prob']}",
        f"- Extra candidates per recoverable GT error: **{summary['extra_candidates_per_gt_err']}**",
        f"- Admit-all unsafe: **{summary['admit_source_top1_unsafe']}**",
        f"- ILP stolen children that are not any GT pair: {summary['ilp_stolen_child_not_gt']} / {buckets['ilp_global']}",
        "",
        "## Ranking-below-gate detail",
        "",
        f"- source_rank=1 (true continuation excluded by gate): {summary['ranking_below_gate_source_top1']}",
        f"- source already has a different gated child: {summary['ranking_below_gate_wrong_gated_child']}",
        f"- source has no gated child: {summary['ranking_below_gate_no_gated_child']}",
        "",
        "## Recoverability (all errors including detection)",
        "",
        "| Lever | Count |",
        "| --- | ---: |",
    ]
    for key in sorted(recover):
        lines.append(f"| {key} | {recover[key]} |")
    lines.extend(
        [
            "",
            f"ILP conflict types: `{summary['ilp_conflict_counts']}`",
            f"Rematch rows sharing a node with another conflict: {summary['rematch_cascade_from_conflict']}",
            "",
            "## Error table",
            "",
            "See `error_table.csv` in the experiment directory.",
            "",
        ]
    )
    if not table.empty:
        show = table[
            [
                c
                for c in [
                    "t",
                    "bucket",
                    "recoverability",
                    "conflict",
                    "target_rank",
                    "source_rank",
                    "blended_prob",
                    "candidate_present",
                    "gt_is_source_top1",
                    "selected_out",
                    "selected_in",
                    "rematch_shares_conflict_node",
                ]
                if c in table.columns
            ]
        ]
        lines.append(show.to_string(index=False))
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = _parser().parse_args()
    capture_root = args.capture_root
    diagnostic = pd.read_csv(capture_root / "candidate_edge_diagnostic.csv")
    geff = args.geff
    submission_path = capture_root / "predictions" / "postprocessed_submission.csv"
    submission = pd.read_csv(submission_path) if submission_path.exists() else None
    table, summary = audit_errors(
        diagnostic,
        capture_root / "candidate_capture" / args.dataset,
        dataset=args.dataset,
        geff_path=geff if geff.exists() else None,
        submission=submission,
    )
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "error_table.csv", index=False)
    (out / "summary.json").write_text(
        json.dumps(_json_safe(summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report = out / "report.md"
    _write_report(report, summary, table)
    analysis = Path("outputs/analysis/oa_ilp_candidate_audit_44b6_12dfb391_report.md")
    analysis.parent.mkdir(parents=True, exist_ok=True)
    analysis.write_text(report.read_text(encoding="utf-8"), encoding="utf-8")
    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
