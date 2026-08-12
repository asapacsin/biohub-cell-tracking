#!/usr/bin/env python3
"""Attribute fresh candidate captures into A–E taxonomy + ranking diagnostics.

Maps candidate_bottleneck causes to user A–E:
  A Candidate/gating     <- detection_miss
  B Ranking              <- scorer_ranking
  C Threshold            <- candidate_threshold
  D Assignment/consistency/postprocess
                         <- ilp_global + postprocessing_removed + postprocessing_rematch
  E Other                <- residual (should be empty)

Focuses on ordinary associations only (already the unit of candidate_bottleneck).
Shares for the ranking hypothesis use endpoint-matched causal errors (exclude
detection_miss), matching prior bottleneck reports.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

CAUSE_TO_AE = {
    "detection_miss": "A_candidate_gating",
    "scorer_ranking": "B_ranking",
    "candidate_threshold": "C_threshold",
    "ilp_global": "D_assignment_consistency_postprocess",
    "postprocessing_removed": "D_assignment_consistency_postprocess",
    "postprocessing_rematch": "D_assignment_consistency_postprocess",
}

AE_ORDER = [
    "A_candidate_gating",
    "B_ranking",
    "C_threshold",
    "D_assignment_consistency_postprocess",
    "E_other",
]

AE_LABELS = {
    "A_candidate_gating": "Candidate/gating",
    "B_ranking": "Ranking",
    "C_threshold": "Threshold",
    "D_assignment_consistency_postprocess": "Assignment/consistency",
    "E_other": "Other",
}

EXPECTED_SCORES = {
    "fixed8": 0.9181439782806684,
    "holdout8": 0.9646726188580379,
}


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--root",
        type=Path,
        default=Path("outputs/experiments/candidate_capture_recipe_c_edge_0_40_v1"),
    )
    p.add_argument(
        "--nfs-root",
        type=Path,
        default=Path.home() / "biohub-outputs/experiments/candidate_capture_recipe_c_edge_0_40_v1",
    )
    return p


def _load_split(split_dir: Path) -> dict[str, Any]:
    diagnostic = pd.read_csv(split_dir / "candidate_edge_diagnostic.csv")
    summary = pd.read_csv(split_dir / "cause_summary.csv")
    decision = json.loads((split_dir / "decision.json").read_text())
    score = json.loads((split_dir / "score_summary.json").read_text())
    metadata = json.loads((split_dir / "metadata.json").read_text())
    config_text = (split_dir / "experiment_config.yaml").read_text()
    return {
        "diagnostic": diagnostic,
        "summary": summary,
        "decision": decision,
        "score": score,
        "metadata": metadata,
        "config_text": config_text,
    }


def _map_ae(cause: str) -> str:
    return CAUSE_TO_AE.get(str(cause), "E_other")


def _ae_counts(errors: pd.DataFrame) -> dict[str, int]:
    mapped = errors["cause"].map(_map_ae)
    return {label: int((mapped == label).sum()) for label in AE_ORDER}


def _ranking_diagnostics(frame: pd.DataFrame) -> dict[str, Any]:
    """Ranking-focused stats on ordinary associations."""
    endpoint = frame[frame["source_detected"] & frame["target_detected"]].copy()
    recorded = endpoint[endpoint["candidate_recorded"]].copy()
    ranking_errors = frame[frame["cause"] == "scorer_ranking"]
    # Conditional: among endpoint-matched ordinary assoc where GT pair was recorded
    # (exists in capture / candidate neighborhood), what fraction fail due to ranking.
    gt_edge_exists = recorded  # recorded means GT pair present in capture union
    ranking_among_recorded_errors = gt_edge_exists[gt_edge_exists["cause"] != "correct"]
    ranking_rate_if_gt_edge = (
        float((gt_edge_exists["cause"] == "scorer_ranking").mean()) if len(gt_edge_exists) else np.nan
    )
    ranking_share_of_recorded_errors = (
        float((ranking_among_recorded_errors["cause"] == "scorer_ranking").mean())
        if len(ranking_among_recorded_errors)
        else np.nan
    )

    ranks = recorded["target_rank"].astype(int)
    rank_hist = {str(k): int(v) for k, v in ranks.value_counts().sort_index().items()}
    # Near-miss: ranking errors at target_rank == 2
    rank2_errors = ranking_errors[ranking_errors["target_rank"] == 2]
    margins = ranking_errors["competitor_margin"].dropna()
    recorded_margins = recorded["competitor_margin"].dropna()

    return {
        "endpoint_matched": int(len(endpoint)),
        "gt_edge_recorded_in_capture": int(len(gt_edge_exists)),
        "gt_edge_recorded_errors": int(len(ranking_among_recorded_errors)),
        "ranking_errors": int(len(ranking_errors)),
        "ranking_error_rate_among_recorded_gt_edges": ranking_rate_if_gt_edge,
        "ranking_share_among_recorded_gt_edge_errors": ranking_share_of_recorded_errors,
        "target_rank_distribution_recorded": rank_hist,
        "target_rank_distribution_ranking_errors": {
            str(k): int(v)
            for k, v in ranking_errors["target_rank"].astype(int).value_counts().sort_index().items()
        },
        "rank2_near_miss_count": int(len(rank2_errors)),
        "rank2_near_miss_share_of_ranking_errors": (
            float(len(rank2_errors) / len(ranking_errors)) if len(ranking_errors) else np.nan
        ),
        "ranking_error_competitor_margin": {
            "count": int(len(margins)),
            "mean": float(margins.mean()) if len(margins) else np.nan,
            "median": float(margins.median()) if len(margins) else np.nan,
            "p25": float(margins.quantile(0.25)) if len(margins) else np.nan,
            "p75": float(margins.quantile(0.75)) if len(margins) else np.nan,
        },
        "recorded_gt_edge_competitor_margin": {
            "count": int(len(recorded_margins)),
            "mean": float(recorded_margins.mean()) if len(recorded_margins) else np.nan,
            "median": float(recorded_margins.median()) if len(recorded_margins) else np.nan,
        },
        "top1_rate_among_recorded": float((ranks == 1).mean()) if len(ranks) else np.nan,
        "top2_rate_among_recorded": float((ranks <= 2).mean()) if len(ranks) else np.nan,
        "top3_rate_among_recorded": float((ranks <= 3).mean()) if len(ranks) else np.nan,
    }


def _sample_rows(frame: pd.DataFrame, cause: str, n: int = 5) -> list[dict[str, Any]]:
    part = frame[frame["cause"] == cause]
    if part.empty:
        return []
    cols = [
        "dataset",
        "t",
        "gt_source_id",
        "gt_target_id",
        "target_rank",
        "blended_prob",
        "best_competing_prob",
        "competitor_margin",
        "candidate_present",
        "ilp_selected",
        "final_pair_present",
        "source_in_final",
        "target_in_final",
    ]
    cols = [c for c in cols if c in part.columns]
    return part.sample(n=min(n, len(part)), random_state=0)[cols].to_dict(orient="records")


def _attribute_split(name: str, data: dict[str, Any]) -> dict[str, Any]:
    frame = data["diagnostic"]
    overall = data["summary"][
        (data["summary"]["group_type"] == "overall") & (data["summary"]["group"] == "all")
    ].iloc[0]
    all_errors = frame[frame["cause"] != "correct"]
    causal_errors = all_errors[all_errors["source_detected"] & all_errors["target_detected"]]

    ae_all = _ae_counts(all_errors)
    ae_causal = _ae_counts(causal_errors)
    n_all = max(int(len(all_errors)), 1)
    n_causal = max(int(len(causal_errors)), 1)

    expected = EXPECTED_SCORES[name]
    got = float(data["score"]["score"])
    score_delta = got - expected

    # Consistency: decision shares vs recomputed
    ranking_share_causal = ae_causal["B_ranking"] / n_causal if len(causal_errors) else 0.0
    decision_ranking = float(data["decision"].get("ranking_share", float("nan")))

    # Capture metadata gate
    gates = []
    for ds_dir_hint in []:
        pass
    # Infer gate from overall summary / first rows metadata not stored per-row;
    # check experiment config text and command.
    motion_off = "output_motion_relink: false" in data["config_text"] or (
        "output_motion_relink: false" in data["config_text"].replace("'", "")
    )
    # YAML may serialize false as false
    motion_off = "output_motion_relink: false" in data["config_text"]
    edge_ok = ("edge_threshold: 0.4" in data["config_text"]) or (
        "edge_threshold: 0.40" in data["config_text"]
    )

    return {
        "split": name,
        "score": got,
        "expected_score": expected,
        "score_delta": score_delta,
        "score_match": abs(score_delta) < 1e-9,
        "recipe_checks": {
            "motion_relink_false_in_config": motion_off,
            "edge_threshold_0_4_in_config": edge_ok,
            "config_path": data["metadata"].get("config"),
            "top_k": data["metadata"].get("top_k"),
        },
        "ordinary_associations": int(overall["ordinary_associations"]),
        "all_errors": int(len(all_errors)),
        "causal_errors": int(len(causal_errors)),
        "native_counts": {
            c: int(overall[c])
            for c in (
                "correct",
                "detection_miss",
                "scorer_ranking",
                "candidate_threshold",
                "ilp_global",
                "postprocessing_removed",
                "postprocessing_rematch",
            )
        },
        "ae_all_errors": {
            AE_LABELS[k]: {"count": ae_all[k], "pct": ae_all[k] / n_all * 100.0}
            for k in AE_ORDER
        },
        "ae_causal_errors": {
            AE_LABELS[k]: {"count": ae_causal[k], "pct": ae_causal[k] / n_causal * 100.0}
            for k in AE_ORDER
            if k != "A_candidate_gating"  # excluded from causal by definition
        },
        # Keep A in causal table as 0 for clarity
        "ae_causal_errors_full": {
            AE_LABELS[k]: {
                "count": ae_causal[k],
                "pct": (ae_causal[k] / n_causal * 100.0) if len(causal_errors) else 0.0,
            }
            for k in AE_ORDER
        },
        "ranking_share_causal": ranking_share_causal,
        "decision_ranking_share": decision_ranking,
        "ranking_share_consistency_ok": abs(ranking_share_causal - decision_ranking) < 1e-9
        if np.isfinite(decision_ranking)
        else False,
        "ranking_diagnostics": _ranking_diagnostics(frame),
        "samples": {
            cause: _sample_rows(frame, cause)
            for cause in (
                "scorer_ranking",
                "candidate_threshold",
                "ilp_global",
                "postprocessing_removed",
                "postprocessing_rematch",
                "detection_miss",
            )
        },
        "decision": data["decision"],
    }


def _combine(fixed: dict[str, Any], hold: dict[str, Any]) -> dict[str, Any]:
    labels = list(AE_LABELS.values())
    combined_all: dict[str, dict[str, float]] = {}
    combined_causal: dict[str, dict[str, float]] = {}
    for label in labels:
        c_all = fixed["ae_all_errors"][label]["count"] + hold["ae_all_errors"][label]["count"]
        c_causal = (
            fixed["ae_causal_errors_full"][label]["count"]
            + hold["ae_causal_errors_full"][label]["count"]
        )
        combined_all[label] = {"count": c_all}
        combined_causal[label] = {"count": c_causal}
    n_all = sum(v["count"] for v in combined_all.values()) or 1
    n_causal = sum(v["count"] for v in combined_causal.values()) or 1
    for label in labels:
        combined_all[label]["pct"] = combined_all[label]["count"] / n_all * 100.0
        combined_causal[label]["pct"] = combined_causal[label]["count"] / n_causal * 100.0

    ranking_fixed = fixed["ranking_share_causal"] * 100.0
    ranking_hold = hold["ranking_share_causal"] * 100.0
    ranking_combined = combined_causal["Ranking"]["pct"]
    # Hypothesis: ranking ~50% of remaining ordinary-association (causal) failures
    supported = all(abs(x - 50.0) <= 10.0 for x in (ranking_fixed, ranking_hold, ranking_combined))
    # More precise: prior claim was ~50-54%; require both splits in [40, 60] and combined ~50
    in_band = (
        40.0 <= ranking_fixed <= 60.0
        and 40.0 <= ranking_hold <= 60.0
        and 40.0 <= ranking_combined <= 60.0
    )
    return {
        "ae_all_errors": combined_all,
        "ae_causal_errors": combined_causal,
        "all_errors": int(n_all),
        "causal_errors": int(n_causal),
        "ranking_pct_fixed8": ranking_fixed,
        "ranking_pct_holdout8": ranking_hold,
        "ranking_pct_combined": ranking_combined,
        "hypothesis_ranking_approx_50pct": {
            "SUPPORTED" if in_band else "NOT_SUPPORTED": True,
            "verdict": "SUPPORTED" if in_band else "NOT_SUPPORTED",
            "rule": "causal ranking share in [40%, 60%] on fixed-8, holdout-8, and combined",
            "note": "Matches prior ~50% claim band; not the stricter 60%/20pp promote rule.",
        },
    }


def _markdown_report(payload: dict[str, Any]) -> str:
    fixed = payload["fixed8"]
    hold = payload["holdout8"]
    comb = payload["combined"]
    verdict = comb["hypothesis_ranking_approx_50pct"]["verdict"]

    def row(label: str) -> str:
        fa = fixed["ae_causal_errors_full"][label]
        ha = hold["ae_causal_errors_full"][label]
        ca = comb["ae_causal_errors"][label]
        return (
            f"| {label} | {fa['count']} ({fa['pct']:.1f}%) | "
            f"{ha['count']} ({ha['pct']:.1f}%) | {ca['pct']:.1f}% |"
        )

    lines = [
        "# Fresh candidate-edge capture attribution (recipe C edge=0.40 motion OFF)",
        "",
        "## Recipe verified",
        f"- motion_relink OFF: fixed={fixed['recipe_checks']['motion_relink_false_in_config']},"
        f" holdout={hold['recipe_checks']['motion_relink_false_in_config']}",
        f"- edge_threshold 0.40: fixed={fixed['recipe_checks']['edge_threshold_0_4_in_config']},"
        f" holdout={hold['recipe_checks']['edge_threshold_0_4_in_config']}",
        f"- config: `{fixed['recipe_checks']['config_path']}`",
        "",
        "## Scores",
        f"- fixed-8: `{fixed['score']:.12f}` (expected `{fixed['expected_score']:.12f}`,"
        f" delta `{fixed['score_delta']:+.3e}`, match={fixed['score_match']})",
        f"- holdout-8: `{hold['score']:.12f}` (expected `{hold['expected_score']:.12f}`,"
        f" delta `{hold['score_delta']:+.3e}`, match={hold['score_match']})",
        "",
        "## Fresh attribution (causal / endpoint-matched ordinary errors)",
        "",
        "| Category | fixed-8 | holdout-8 | combined % |",
        "| --- | --- | --- | --- |",
    ]
    for label in AE_LABELS.values():
        lines.append(row(label))
    lines += [
        "",
        f"Causal errors: fixed={fixed['causal_errors']}, holdout={hold['causal_errors']},"
        f" combined={comb['causal_errors']}",
        "",
        "## Ranking diagnostics",
        "",
        "### Fixed-8",
        f"- GT rank distribution (recorded): `{fixed['ranking_diagnostics']['target_rank_distribution_recorded']}`",
        f"- Ranking-error ranks: `{fixed['ranking_diagnostics']['target_rank_distribution_ranking_errors']}`",
        f"- Rank-2 near misses: {fixed['ranking_diagnostics']['rank2_near_miss_count']}"
        f" ({fixed['ranking_diagnostics']['rank2_near_miss_share_of_ranking_errors']:.1%} of ranking errors)",
        f"- Ranking-error score-margin median: "
        f"{fixed['ranking_diagnostics']['ranking_error_competitor_margin']['median']}",
        f"- Ranking-error rate among recorded GT edges: "
        f"{fixed['ranking_diagnostics']['ranking_error_rate_among_recorded_gt_edges']:.2%}",
        "",
        "### Holdout-8",
        f"- GT rank distribution (recorded): `{hold['ranking_diagnostics']['target_rank_distribution_recorded']}`",
        f"- Ranking-error ranks: `{hold['ranking_diagnostics']['target_rank_distribution_ranking_errors']}`",
        f"- Rank-2 near misses: {hold['ranking_diagnostics']['rank2_near_miss_count']}"
        f" ({hold['ranking_diagnostics']['rank2_near_miss_share_of_ranking_errors']:.1%} of ranking errors)",
        f"- Ranking-error score-margin median: "
        f"{hold['ranking_diagnostics']['ranking_error_competitor_margin']['median']}",
        f"- Ranking-error rate among recorded GT edges: "
        f"{hold['ranking_diagnostics']['ranking_error_rate_among_recorded_gt_edges']:.2%}",
        "",
        "## Conclusion",
        f"**{verdict}** that ranking accounts for ~50% of remaining ordinary-association "
        f"(causal) failures under this recipe.",
        f"- fixed-8 ranking share: **{comb['ranking_pct_fixed8']:.1f}%**",
        f"- holdout-8 ranking share: **{comb['ranking_pct_holdout8']:.1f}%**",
        f"- combined ranking share: **{comb['ranking_pct_combined']:.1f}%**",
        "",
        "## Recommended next experiment",
        payload["recommended_next_experiment"],
        "",
        f"NFS root: `{payload['nfs_root']}`",
        f"Login mirror: `{payload['login_root']}`",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    args = _parser().parse_args()
    login_root = args.root
    nfs_root = args.nfs_root

    # Prefer login mirror; fall back to NFS if needed
    def resolve(split: str) -> Path:
        local = login_root / split
        if (local / "candidate_edge_diagnostic.csv").exists():
            return local
        remote = nfs_root / split
        if (remote / "candidate_edge_diagnostic.csv").exists():
            return remote
        raise FileNotFoundError(f"missing diagnostic for {split} under {local} or {remote}")

    fixed = _attribute_split("fixed8", _load_split(resolve("fixed8")))
    hold = _attribute_split("holdout8", _load_split(resolve("holdout8")))
    combined = _combine(fixed, hold)

    recommended = (
        "One targeted edge-scorer ranking improvement (retrain or re-rank) evaluated on "
        "fixed-8 + holdout-8 under frozen recipe "
        "`recipe_c_motion_off_edge_0_40_det0_96875` (motion OFF, edge=0.40); "
        "do not reopen short-track OFF or further edge-gate sweeps."
        if combined["hypothesis_ranking_approx_50pct"]["verdict"] == "SUPPORTED"
        else "Re-examine the largest non-ranking causal bucket on these fresh captures "
        "(assignment/consistency/postprocess vs threshold) with one targeted ablation; "
        "keep the promoted recipe frozen."
    )

    payload = {
        "schema_version": 1,
        "experiment": "candidate_capture_recipe_c_edge_0_40_v1",
        "login_root": str(login_root),
        "nfs_root": str(nfs_root),
        "taxonomy_map": {
            "A_candidate_gating": "detection_miss",
            "B_ranking": "scorer_ranking",
            "C_threshold": "candidate_threshold",
            "D_assignment_consistency_postprocess": (
                "ilp_global + postprocessing_removed + postprocessing_rematch"
            ),
            "E_other": "residual",
        },
        "fixed8": fixed,
        "holdout8": hold,
        "combined": combined,
        "recommended_next_experiment": recommended,
    }

    out = login_root
    out.mkdir(parents=True, exist_ok=True)
    (out / "fresh_attribution.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n"
    )
    report = _markdown_report(payload)
    (out / "fresh_attribution_report.md").write_text(report)
    # Also write analysis copy
    analysis = Path("outputs/analysis")
    analysis.mkdir(parents=True, exist_ok=True)
    (analysis / "fresh_candidate_capture_edge_0_40_report.md").write_text(report)
    print(json.dumps({"verdict": combined["hypothesis_ranking_approx_50pct"]["verdict"],
                      "ranking_pct": {
                          "fixed8": combined["ranking_pct_fixed8"],
                          "holdout8": combined["ranking_pct_holdout8"],
                          "combined": combined["ranking_pct_combined"],
                      },
                      "scores": {"fixed8": fixed["score"], "holdout8": hold["score"]},
                      "score_match": {
                          "fixed8": fixed["score_match"],
                          "holdout8": hold["score_match"],
                      }}, indent=2))


if __name__ == "__main__":
    main()
