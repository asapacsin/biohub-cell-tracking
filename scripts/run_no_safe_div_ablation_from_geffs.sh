#!/usr/bin/env bash
# Re-convert saved two-seed det=0.960 GEFFs with safe-division postlink OFF (no re-inference).
set -euo pipefail

DEST="${DEST:-/tmp/${USER}/biohub-no-safe-div-ablation}"
CONTROL_RAW="${CONTROL_RAW:-${HOME}/biohub-outputs/fixed8/two_seed_det_thresh_0_960/raw_geff}"
OUT="${OUT:-${HOME}/biohub-outputs/fixed8/two_seed_det0_960_no_safe_div}"
CONTROL_SUMMARY="${CONTROL_SUMMARY:-${HOME}/biohub-outputs/fixed8/two_seed_det_thresh_0_960/summary.json}"
CONTROL_PER="${CONTROL_PER:-${HOME}/biohub-outputs/fixed8/two_seed_det_thresh_0_960/per_dataset.csv}"
CONTROL_PRED="${CONTROL_PRED:-${HOME}/biohub-outputs/fixed8/two_seed_det_thresh_0_960/predictions/postprocessed_submission.csv}"
DC_SUMMARY="${DC_SUMMARY:-${HOME}/biohub-outputs/fixed8/two_seed_det0_960_deepcenter/summary.json}"
CONFIG_REL="configs/experiments/two_seed_det0_960_no_safe_div.yaml"
CONTROL_SCORE="${CONTROL_SCORE:-0.8879786102989093}"
DC_SCORE="${DC_SCORE:-0.8883300756289333}"

cd "$DEST"
source "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate biohub
export PYTHONPATH="${DEST}/src"

test -d "$CONTROL_RAW"
test "$(find "$CONTROL_RAW" -maxdepth 1 -name '*.geff' | wc -l)" -eq 8
test -f "$CONFIG_REL"

rm -rf "$OUT"
mkdir -p "$OUT/predictions" "$OUT/config"
cp -a "$CONFIG_REL" "$OUT/config/two_seed_det0_960_no_safe_div.yaml"

CONTROL_SCORE="$CONTROL_SCORE" DC_SCORE="$DC_SCORE" \
CONTROL_RAW="$CONTROL_RAW" OUT="$OUT" CONFIG_REL="$CONFIG_REL" \
CONTROL_SUMMARY="$CONTROL_SUMMARY" CONTROL_PER="$CONTROL_PER" \
CONTROL_PRED="$CONTROL_PRED" DC_SUMMARY="$DC_SUMMARY" \
python - <<'PY'
from __future__ import annotations

import csv
import json
import os
import shutil
import time
from collections import defaultdict
from pathlib import Path

import tracksdata as td

from biohub_pipeline import postprocessing
from biohub_pipeline.config import load_config
from biohub_pipeline.fixed8_cv import (
    MATERIAL_DELTA,
    evaluate_postprocessed_predictions,
    write_metric_outputs,
)
from biohub_pipeline.submission import graph_rows, validate_submission_file, write_rows

control_score = float(os.environ["CONTROL_SCORE"])
dc_score = float(os.environ["DC_SCORE"])
control_raw = Path(os.environ["CONTROL_RAW"])
out = Path(os.environ["OUT"])
config_path = Path(os.environ["CONFIG_REL"])
control_summary_path = Path(os.environ["CONTROL_SUMMARY"])
control_per_path = Path(os.environ["CONTROL_PER"])
control_pred_path = Path(os.environ["CONTROL_PRED"])
dc_summary_path = Path(os.environ["DC_SUMMARY"])
data_dir = Path("data/competition/train")

config = load_config(config_path)
assert float(config.inference["detection_threshold"]) == 0.960
assert float(config.inference["ensemble_alpha"]) == 0.5
assert config.postprocessing["output_safe_divisions"] is False
assert config.postprocessing["output_gap2_recovery"] is False
assert config.postprocessing["use_deepcenter_veto"] is False
assert config.postprocessing["deepcenter_gap_veto"] is False
assert config.postprocessing["deepcenter_safe_div_veto"] is False

raw_dst = out / "raw_geff_source"
raw_dst.mkdir(parents=True, exist_ok=True)
for src in sorted(control_raw.glob("*.geff")):
    dest = raw_dst / src.name
    if src.is_dir():
        shutil.copytree(src, dest)
    else:
        shutil.copy2(src, dest)

# Also run control-equivalent postprocess (safe_div ON) on the same GEFFs to
# identify exact safe-division edges and verify ordinary edges match.
control_cfg = load_config(Path("configs/sweeps/two_seed_det_thresh_0_960.yaml"))
assert control_cfg.postprocessing["output_safe_divisions"] is True

postprocessing.configure(config.postprocessing, data_dir)

started = time.perf_counter()
all_rows: list[dict] = []
stats_by_dataset: dict[str, dict] = {}
safe_div_edges_control: list[dict] = []
ordinary_identity: dict[str, dict] = {}

for geff_path in sorted(raw_dst.glob("*.geff")):
    dataset = geff_path.stem
    loaded = td.graph.IndexedRXGraph.from_geff(geff_path)
    graph = loaded[0] if isinstance(loaded, tuple) else loaded
    nodes = {
        int(row["node_id"]): {
            "node_id": int(row["node_id"]),
            "t": int(row["t"]),
            "z": float(row["z"]),
            "y": float(row["y"]),
            "x": float(row["x"]),
        }
        for row in graph.node_attrs().iter_rows(named=True)
    }
    edges = []
    for row in graph.edge_attrs().iter_rows(named=True):
        probability = row.get("edge_prob") if hasattr(row, "get") else None
        edges.append(
            {
                "source_id": int(row["source_id"]),
                "target_id": int(row["target_id"]),
                "edge_prob": None if probability is None else float(probability),
            }
        )

    # Experiment: safe_div OFF
    postprocessing.configure(config.postprocessing, data_dir)
    nodes_off, edges_off, stats_off = postprocessing.filter_output_graph(
        {k: dict(v) for k, v in nodes.items()},
        [dict(e) for e in edges],
        dataset=dataset,
    )
    assert int(stats_off.get("safe_divisions_added", 0)) == 0
    assert int(stats_off.get("safe_division_candidates", 0)) == 0

    # Control postprocess on same raw graph for identity check
    postprocessing.configure(control_cfg.postprocessing, data_dir)
    nodes_on, edges_on, stats_on = postprocessing.filter_output_graph(
        {k: dict(v) for k, v in nodes.items()},
        [dict(e) for e in edges],
        dataset=dataset,
    )

    on_keys = {(int(e["source_id"]), int(e["target_id"])) for e in edges_on}
    off_keys = {(int(e["source_id"]), int(e["target_id"])) for e in edges_off}
    safe_keys = {
        (int(e["source_id"]), int(e["target_id"]))
        for e in edges_on
        if int(e.get("safe_division", 0) or 0) == 1
    }
    ordinary_on = on_keys - safe_keys
    only_in_off = off_keys - on_keys
    ordinary_missing_in_off = ordinary_on - off_keys
    extra_non_safe_in_on = (on_keys - off_keys) - safe_keys

    for e in edges_on:
        if int(e.get("safe_division", 0) or 0) == 1:
            safe_div_edges_control.append(
                {
                    "dataset": dataset,
                    "source_id": int(e["source_id"]),
                    "target_id": int(e["target_id"]),
                    "distance_um": e.get("distance_um"),
                }
            )

    ordinary_identity[dataset] = {
        "n_edges_control": len(on_keys),
        "n_edges_no_safe_div": len(off_keys),
        "n_safe_division_edges_in_control": len(safe_keys),
        "n_ordinary_edges_control": len(ordinary_on),
        "ordinary_edges_identical": ordinary_on == off_keys and not only_in_off and not ordinary_missing_in_off,
        "n_ordinary_missing_in_no_safe_div": len(ordinary_missing_in_off),
        "n_extra_edges_in_no_safe_div": len(only_in_off),
        "n_non_safe_edges_removed_unexpectedly": len(extra_non_safe_in_on),
        "safe_division_candidates": int(stats_on.get("safe_division_candidates", 0)),
        "safe_divisions_added": int(stats_on.get("safe_divisions_added", 0)),
    }
    stats_by_dataset[dataset] = {
        "safe_division_candidates_control": int(stats_on.get("safe_division_candidates", 0)),
        "safe_divisions_added_control": int(stats_on.get("safe_divisions_added", 0)),
        "safe_divisions_added_experiment": int(stats_off.get("safe_divisions_added", 0)),
    }

    all_rows.extend(graph_rows(dataset, nodes_off, edges_off, len(all_rows)))

pred_csv = out / "predictions" / "postprocessed_submission.csv"
write_rows(all_rows, pred_csv)
print("CONVERT_OK", validate_submission_file(pred_csv))

rows, aggregate = evaluate_postprocessed_predictions(pred_csv, data_dir, config)
runtime = time.perf_counter() - started
delta = float(aggregate["score"]) - control_score
delta_vs_dc = float(aggregate["score"]) - dc_score

control_summary = json.loads(control_summary_path.read_text())
dc_summary = json.loads(dc_summary_path.read_text()) if dc_summary_path.is_file() else {
    "score": dc_score, "edge_tp": 3877, "edge_fp": 263, "edge_fn": 226,
    "division_tp": 0, "division_fp": 0, "division_fn": 7,
}

ordinary_all_identical = all(v["ordinary_edges_identical"] for v in ordinary_identity.values())
n_safe_removed = sum(v["n_safe_division_edges_in_control"] for v in ordinary_identity.values())
n_safe_candidates = sum(v["safe_division_candidates_control"] for v in stats_by_dataset.values())

summary = {
    **aggregate,
    "detection_threshold": 0.960,
    "ensemble_alpha": 0.5,
    "output_safe_divisions": False,
    "output_gap2_recovery": False,
    "use_deepcenter_veto": False,
    "checkpoint_path": "reuse_two_seed_det0_960_raw_geff",
    "runtime_seconds": runtime,
    "control_score": control_score,
    "delta_vs_control": delta,
    "deepcenter_score": float(dc_summary["score"]),
    "delta_vs_deepcenter": delta_vs_dc,
    "control_difference_material": abs(delta) > MATERIAL_DELTA,
    "safe_division_candidates_control_total": n_safe_candidates,
    "safe_division_edges_removed_total": n_safe_removed,
    "ordinary_edges_identical_to_control": ordinary_all_identical,
}
manifest = {
    "schema_version": 1,
    "experiment": "two_seed_det0_960_no_safe_div",
    "config": str(config_path),
    "control_raw_geff": str(control_raw),
    "note": "Safe-division postlink OFF only; DeepCenter/gap2 off; no re-inference; control untouched",
}
write_metric_outputs(out, rows, summary, manifest)

control_per = {
    row["dataset"]: float(row["adj_edge_jaccard"])
    for row in csv.DictReader(control_per_path.open(newline="", encoding="utf-8"))
}
compare_rows = []
n_improved = n_worsened = n_unchanged = 0
severe = []
for row in rows:
    ds = row["dataset"]
    score = float(row["adj_edge_jaccard"])
    base = control_per[ds]
    d = score - base
    if d > 1e-12:
        n_improved += 1
    elif d < -1e-12:
        n_worsened += 1
    else:
        n_unchanged += 1
    if d < -0.005:
        severe.append({"dataset": ds, "delta": d})
    compare_rows.append(
        {
            "dataset": ds,
            "control_score": base,
            "no_safe_div_score": score,
            "delta": d,
            "edge_tp": row["edge_tp"],
            "edge_fp": row["edge_fp"],
            "edge_fn": row["edge_fn"],
            "division_tp": row["division_tp"],
            "division_fp": row["division_fp"],
            "division_fn": row["division_fn"],
        }
    )

edge_delta = {
    "tp": int(summary["edge_tp"]) - int(control_summary["edge_tp"]),
    "fp": int(summary["edge_fp"]) - int(control_summary["edge_fp"]),
    "fn": int(summary["edge_fn"]) - int(control_summary["edge_fn"]),
}
div_delta = {
    "tp": int(summary["division_tp"]) - int(control_summary["division_tp"]),
    "fp": int(summary["division_fp"]) - int(control_summary["division_fp"]),
    "fn": int(summary["division_fn"]) - int(control_summary["division_fn"]),
}
edge_delta_vs_dc = {
    "tp": int(summary["edge_tp"]) - int(dc_summary["edge_tp"]),
    "fp": int(summary["edge_fp"]) - int(dc_summary["edge_fp"]),
    "fn": int(summary["edge_fn"]) - int(dc_summary["edge_fn"]),
}
div_delta_vs_dc = {
    "tp": int(summary["division_tp"]) - int(dc_summary["division_tp"]),
    "fp": int(summary["division_fp"]) - int(dc_summary["division_fp"]),
    "fn": int(summary["division_fn"]) - int(dc_summary["division_fn"]),
}

# Interpretation vs DeepCenter / new-base rule
prefer_no_safe_div = (
    float(summary["score"]) >= dc_score - 1e-12
    and int(summary["edge_tp"]) >= int(control_summary["edge_tp"]) - 2
)
new_base = (
    int(summary["division_fp"]) == 0
    and abs(edge_delta["tp"]) <= 2
    and abs(edge_delta["fp"]) <= 2
    and abs(edge_delta["fn"]) <= 2
    and delta >= -1e-6
)
interpretation = {
    "prefer_no_safe_div_over_deepcenter": prefer_no_safe_div,
    "becomes_new_base": new_base,
    "ordinary_edges_identical_to_control": ordinary_all_identical,
    "notes": [],
}
if prefer_no_safe_div:
    interpretation["notes"].append(
        "no_safe_div matches/exceeds DeepCenter score while retaining control-level edge TP"
    )
if new_base:
    interpretation["notes"].append(
        "division FP 15→0 with essentially unchanged ordinary edge metrics → new base candidate"
    )
if not ordinary_all_identical:
    interpretation["notes"].append(
        "WARNING: ordinary edges differ from control; inspect parent/child construction"
    )
if delta < 0:
    interpretation["notes"].append("score fell vs control")

decision = {
    "delta_vs_control": delta,
    "delta_vs_deepcenter": delta_vs_dc,
    "n_improved": n_improved,
    "n_unchanged": n_unchanged,
    "n_worsened": n_worsened,
    "edge_delta_vs_control": edge_delta,
    "division_delta_vs_control": div_delta,
    "edge_delta_vs_deepcenter": edge_delta_vs_dc,
    "division_delta_vs_deepcenter": div_delta_vs_dc,
    "severe_regressions": severe,
    "interpretation": interpretation,
    "verdict": (
        "NEW_BASE"
        if new_base
        else ("PREFER_OVER_DEEPCENTER" if prefer_no_safe_div else ("REGRESSION" if delta < 0 else "MIXED"))
    ),
}

(out / "ordinary_edge_identity.json").write_text(
    json.dumps(
        {
            "all_identical": ordinary_all_identical,
            "by_dataset": ordinary_identity,
            "safe_division_edges_removed_total": n_safe_removed,
            "safe_division_candidates_control_total": n_safe_candidates,
        },
        indent=2,
        sort_keys=True,
    )
    + "\n"
)
with (out / "safe_division_edges_removed.csv").open("w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=["dataset", "source_id", "target_id", "distance_um"])
    w.writeheader()
    w.writerows(safe_div_edges_control)
(out / "safe_div_postprocess_stats.json").write_text(
    json.dumps(stats_by_dataset, indent=2, sort_keys=True) + "\n"
)
with (out / "per_dataset_vs_control.csv").open("w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=list(compare_rows[0].keys()))
    w.writeheader()
    w.writerows(sorted(compare_rows, key=lambda r: r["dataset"]))

comparison = {
    "control_score": control_score,
    "no_safe_div_score": float(summary["score"]),
    "deepcenter_score": float(dc_summary["score"]),
    "delta_vs_control": delta,
    "delta_vs_deepcenter": delta_vs_dc,
    "control_edges": {"tp": 3885, "fp": 273, "fn": 218},
    "no_safe_div_edges": {
        "tp": int(summary["edge_tp"]),
        "fp": int(summary["edge_fp"]),
        "fn": int(summary["edge_fn"]),
    },
    "deepcenter_edges": {
        "tp": int(dc_summary["edge_tp"]),
        "fp": int(dc_summary["edge_fp"]),
        "fn": int(dc_summary["edge_fn"]),
    },
    "control_divisions": {"tp": 0, "fp": 15, "fn": 7},
    "no_safe_div_divisions": {
        "tp": int(summary["division_tp"]),
        "fp": int(summary["division_fp"]),
        "fn": int(summary["division_fn"]),
    },
    "deepcenter_divisions": {
        "tp": int(dc_summary["division_tp"]),
        "fp": int(dc_summary["division_fp"]),
        "fn": int(dc_summary["division_fn"]),
    },
    "per_dataset": compare_rows,
    "ordinary_edge_identity": ordinary_identity,
    "decision": decision,
}
(out / "comparison_vs_control.json").write_text(
    json.dumps(comparison, indent=2, sort_keys=True) + "\n"
)
(out / "DONE").write_text("ok\n", encoding="utf-8")

print(f"score:      {float(summary['score']):.12f}")
print(f"control:    {control_score:.12f}")
print(f"delta:      {delta:+.12f}")
print(f"deepcenter: {float(dc_summary['score']):.12f}")
print(f"delta_vs_dc:{delta_vs_dc:+.12f}")
print(
    "edges tp/fp/fn:",
    f"{summary['edge_tp']} / {summary['edge_fp']} / {summary['edge_fn']}",
)
print(
    "div tp/fp/fn:",
    f"{summary['division_tp']} / {summary['division_fp']} / {summary['division_fn']}",
)
print(f"datasets improved/unchanged/worsened: {n_improved}/{n_unchanged}/{n_worsened}")
print(f"safe_div candidates/edges removed: {n_safe_candidates}/{n_safe_removed}")
print(f"ordinary_edges_identical: {ordinary_all_identical}")
print("severe:", severe)
print(f"VERDICT {decision['verdict']}")
print("OUT", out)
PY

echo NO_SAFE_DIV_ABLATION_DONE
