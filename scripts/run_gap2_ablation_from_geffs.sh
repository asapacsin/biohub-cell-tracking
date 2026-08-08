#!/usr/bin/env bash
# Re-convert saved two-seed det=0.960 GEFFs with gap-2 recovery ON (no re-inference).
set -euo pipefail

DEST="${DEST:-/tmp/${USER}/biohub-gap2-ablation}"
CONTROL_RAW="${CONTROL_RAW:-${HOME}/biohub-outputs/fixed8/two_seed_det_thresh_0_960/raw_geff}"
OUT="${OUT:-${HOME}/biohub-outputs/fixed8/two_seed_det0_960_gap2}"
CONTROL_SUMMARY="${CONTROL_SUMMARY:-${HOME}/biohub-outputs/fixed8/two_seed_det_thresh_0_960/summary.json}"
CONTROL_PER="${CONTROL_PER:-${HOME}/biohub-outputs/fixed8/two_seed_det_thresh_0_960/per_dataset.csv}"
CONFIG_REL="configs/experiments/two_seed_det0_960_gap2.yaml"
CONTROL_SCORE="${CONTROL_SCORE:-0.8879786102989093}"

cd "$DEST"
source "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate biohub
export PYTHONPATH="${DEST}/src"

test -d "$CONTROL_RAW"
test "$(find "$CONTROL_RAW" -maxdepth 1 -name '*.geff' | wc -l)" -eq 8
test -f "$CONFIG_REL"

rm -rf "$OUT"
mkdir -p "$OUT/predictions" "$OUT/config"

cp -a "$CONFIG_REL" "$OUT/config/two_seed_det0_960_gap2.yaml"

python - <<PY
from __future__ import annotations

import csv
import json
import shutil
import time
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

control_score = float("${CONTROL_SCORE}")
control_raw = Path("${CONTROL_RAW}")
out = Path("${OUT}")
config_path = Path("${CONFIG_REL}")
control_summary_path = Path("${CONTROL_SUMMARY}")
control_per_path = Path("${CONTROL_PER}")
data_dir = Path("data/competition/train")

config = load_config(config_path)
assert float(config.inference["detection_threshold"]) == 0.960
assert float(config.inference["ensemble_alpha"]) == 0.5
assert config.postprocessing["output_gap2_recovery"] is True
assert config.postprocessing["deepcenter_gap_veto"] is False
assert config.postprocessing["deepcenter_safe_div_veto"] is False
assert config.postprocessing["use_deepcenter_veto"] is False

# Preserve control raw GEFFs by copying into experiment dir (do not modify control).
raw_dst = out / "raw_geff_source"
raw_dst.mkdir(parents=True, exist_ok=True)
geffs = sorted(control_raw.glob("*.geff"))
assert len(geffs) == 8
for src in geffs:
    dest = raw_dst / src.name
    if src.is_dir():
        shutil.copytree(src, dest)
    else:
        shutil.copy2(src, dest)

postprocessing.configure(config.postprocessing, data_dir)

started = time.perf_counter()
all_rows: list[dict] = []
gap2_events: list[dict] = []
gap2_edges_detail: list[dict] = []
post_stats_by_dataset: dict[str, dict] = {}

for geff_path in sorted(raw_dst.glob("*.geff")):
    dataset = geff_path.stem
    loaded = td.graph.IndexedRXGraph.from_geff(geff_path)
    graph = loaded[0] if isinstance(loaded, tuple) else loaded
    nodes = {}
    for row in graph.node_attrs().iter_rows(named=True):
        node_id = int(row["node_id"])
        nodes[node_id] = {
            "node_id": node_id,
            "t": int(row["t"]),
            "z": float(row["z"]),
            "y": float(row["y"]),
            "x": float(row["x"]),
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
    before_node_ids = set(nodes)
    before_edge_keys = {(int(e["source_id"]), int(e["target_id"])) for e in edges}
    nodes_out, edges_out, stats = postprocessing.filter_output_graph(
        nodes, edges, dataset=dataset
    )
    post_stats_by_dataset[dataset] = {
        k: int(v) for k, v in stats.items() if str(k).startswith("gap2_")
    }

    # Reconstruct gap-2 bridge events from flagged edges + inserted nodes.
    gap2_edges = [e for e in edges_out if int(e.get("gap2_recovered", 0) or 0) == 1]
    new_nodes = sorted(set(nodes_out) - before_node_ids)
    for edge in gap2_edges:
        src = int(edge["source_id"])
        tgt = int(edge["target_id"])
        gap2_edges_detail.append(
            {
                "dataset": dataset,
                "source_id": src,
                "target_id": tgt,
                "source_t": int(nodes_out[src]["t"]) if src in nodes_out else None,
                "target_t": int(nodes_out[tgt]["t"]) if tgt in nodes_out else None,
                "distance_um": edge.get("distance_um"),
                "source_is_new": src in set(new_nodes),
                "target_is_new": tgt in set(new_nodes),
            }
        )

    # Group into end->start bridges of length 3 edges (t -> t+3).
    from collections import defaultdict

    out_map = defaultdict(list)
    for edge in gap2_edges:
        out_map[int(edge["source_id"])].append(int(edge["target_id"]))
    # Find chain starts: gap2 edge whose source is an original node with no gap2 incoming
    gap2_targets = {int(e["target_id"]) for e in gap2_edges}
    for edge in gap2_edges:
        start = int(edge["source_id"])
        if start in gap2_targets:
            continue
        if start not in before_node_ids:
            continue
        # walk 3 hops
        path = [start]
        cur = start
        ok = True
        for _ in range(3):
            nxts = out_map.get(cur, [])
            if len(nxts) != 1:
                ok = False
                break
            cur = nxts[0]
            path.append(cur)
        if not ok or len(path) != 4:
            continue
        end_id = path[0]
        start_id = path[3]
        if start_id not in before_node_ids:
            continue
        gap2_events.append(
            {
                "dataset": dataset,
                "end_id": end_id,
                "start_id": start_id,
                "end_t": int(nodes_out[end_id]["t"]),
                "start_t": int(nodes_out[start_id]["t"]),
                "inserted_node_ids": path[1:3],
                "edge_path": [
                    {"source_id": path[i], "target_id": path[i + 1]} for i in range(3)
                ],
                "total_distance_um": sum(
                    float(e.get("distance_um") or 0.0)
                    for e in gap2_edges
                    if (int(e["source_id"]), int(e["target_id"]))
                    in {(path[i], path[i + 1]) for i in range(3)}
                ),
            }
        )

    all_rows.extend(graph_rows(dataset, nodes_out, edges_out, len(all_rows)))

pred_csv = out / "predictions" / "postprocessed_submission.csv"
write_rows(all_rows, pred_csv)
counts = validate_submission_file(pred_csv)
print("CONVERT_OK", counts)

rows, aggregate = evaluate_postprocessed_predictions(pred_csv, data_dir, config)
runtime = time.perf_counter() - started
delta = float(aggregate["score"]) - control_score
summary = {
    **aggregate,
    "detection_threshold": float(config.inference["detection_threshold"]),
    "ensemble_alpha": float(config.inference["ensemble_alpha"]),
    "output_gap2_recovery": True,
    "checkpoint_path": "reuse_two_seed_det0_960_raw_geff",
    "runtime_seconds": runtime,
    "control_score": control_score,
    "delta_vs_control": delta,
    "control_difference_material": abs(delta) > MATERIAL_DELTA,
    "gap2_pairs_total": sum(v.get("gap2_pairs_selected", 0) for v in post_stats_by_dataset.values()),
    "gap2_added_edges_total": sum(v.get("gap2_added_edges", 0) for v in post_stats_by_dataset.values()),
    "gap2_added_nodes_total": sum(v.get("gap2_added_nodes", 0) for v in post_stats_by_dataset.values()),
    "gap2_candidates_total": sum(v.get("gap2_candidates", 0) for v in post_stats_by_dataset.values()),
}
manifest = {
    "schema_version": 1,
    "experiment": "two_seed_det0_960_gap2",
    "config": str(config_path),
    "control_raw_geff": str(control_raw),
    "control_summary": str(control_summary_path),
    "note": "Gap-2 recovery only; no re-inference; DeepCenter off; control untouched",
    "artifacts": {
        "summary": "summary.json",
        "per_dataset": "per_dataset.csv",
        "gap2_events": "gap2_recovered_events.json",
        "gap2_edges": "gap2_recovered_edges.csv",
        "post_stats": "gap2_postprocess_stats.json",
        "predictions": "predictions/postprocessed_submission.csv",
        "config_copy": "config/two_seed_det0_960_gap2.yaml",
    },
}
write_metric_outputs(out, rows, summary, manifest)

# Per-dataset vs control
control_per = {}
with control_per_path.open(newline="", encoding="utf-8") as fh:
    for row in csv.DictReader(fh):
        control_per[row["dataset"]] = float(row["adj_edge_jaccard"])

compare_rows = []
n_improved = n_worsened = n_unchanged = 0
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
    compare_rows.append(
        {
            "dataset": ds,
            "control_score": base,
            "gap2_score": score,
            "delta": d,
            "edge_tp": row["edge_tp"],
            "edge_fp": row["edge_fp"],
            "edge_fn": row["edge_fn"],
            "division_tp": row["division_tp"],
            "division_fp": row["division_fp"],
            "division_fn": row["division_fn"],
        }
    )

control_summary = json.loads(control_summary_path.read_text())
decision = {
    "promote_if": "delta >= +0.001 and preferably >=5/8 datasets improved; reject if one-dataset or FP spike",
    "delta": delta,
    "n_improved": n_improved,
    "n_unchanged": n_unchanged,
    "n_worsened": n_worsened,
    "edge_delta": {
        "tp": int(summary["edge_tp"]) - int(control_summary["edge_tp"]),
        "fp": int(summary["edge_fp"]) - int(control_summary["edge_fp"]),
        "fn": int(summary["edge_fn"]) - int(control_summary["edge_fn"]),
    },
    "division_delta": {
        "tp": int(summary["division_tp"]) - int(control_summary["division_tp"]),
        "fp": int(summary["division_fp"]) - int(control_summary["division_fp"]),
        "fn": int(summary["division_fn"]) - int(control_summary["division_fn"]),
    },
}
# Decision rule
promote = False
reject_reason = None
interesting = False
if delta >= 0.001 and n_improved >= 5:
    promote = True
elif delta >= 0.001 and n_improved >= 4 and decision["edge_delta"]["fp"] <= 0 and decision["edge_delta"]["fn"] < 0:
    promote = True
    interesting = True
elif abs(max(compare_rows, key=lambda r: abs(r["delta"]))["delta"]) >= 0.9 * abs(delta) and n_improved <= 2:
    reject_reason = "gain concentrated in too few datasets"
    promote = False
elif decision["edge_delta"]["fp"] >= 10 and delta < 0.002:
    reject_reason = "substantial FP increase"
    promote = False
elif delta >= 0.001:
    # borderline: check concentration
    top = max(compare_rows, key=lambda r: r["delta"])
    if top["delta"] >= 0.8 * delta and n_improved <= 2:
        reject_reason = "gain almost entirely from one/few datasets"
        promote = False
    else:
        interesting = True
        promote = n_improved >= 4 and decision["edge_delta"]["fp"] <= 0
elif delta > 0 and decision["edge_delta"]["fp"] < 0 and decision["edge_delta"]["fn"] < 0:
    interesting = True
    promote = False
    reject_reason = "positive but below +0.001 promote threshold"
else:
    promote = False
    reject_reason = "insufficient gain or regression"

decision.update(
    {
        "promote": promote,
        "interesting": interesting,
        "reject_reason": reject_reason,
        "verdict": "PROMOTE" if promote else ("INTERESTING" if interesting else "REJECT"),
    }
)

(out / "gap2_recovered_events.json").write_text(
    json.dumps(
        {
            "n_events": len(gap2_events),
            "n_gap2_edges": len(gap2_edges_detail),
            "events": gap2_events,
        },
        indent=2,
        sort_keys=True,
    )
    + "\n"
)
with (out / "gap2_recovered_edges.csv").open("w", newline="", encoding="utf-8") as fh:
    fields = [
        "dataset",
        "source_id",
        "target_id",
        "source_t",
        "target_t",
        "distance_um",
        "source_is_new",
        "target_is_new",
    ]
    w = csv.DictWriter(fh, fieldnames=fields)
    w.writeheader()
    w.writerows(gap2_edges_detail)

(out / "gap2_postprocess_stats.json").write_text(
    json.dumps(post_stats_by_dataset, indent=2, sort_keys=True) + "\n"
)
with (out / "per_dataset_vs_control.csv").open("w", newline="", encoding="utf-8") as fh:
    fields = list(compare_rows[0].keys())
    w = csv.DictWriter(fh, fieldnames=fields)
    w.writeheader()
    w.writerows(sorted(compare_rows, key=lambda r: r["dataset"]))

comparison = {
    "control_score": control_score,
    "gap2_score": float(summary["score"]),
    "delta": delta,
    "control_edges": {
        "tp": int(control_summary["edge_tp"]),
        "fp": int(control_summary["edge_fp"]),
        "fn": int(control_summary["edge_fn"]),
    },
    "gap2_edges": {
        "tp": int(summary["edge_tp"]),
        "fp": int(summary["edge_fp"]),
        "fn": int(summary["edge_fn"]),
    },
    "control_divisions": {
        "tp": int(control_summary["division_tp"]),
        "fp": int(control_summary["division_fp"]),
        "fn": int(control_summary["division_fn"]),
    },
    "gap2_divisions": {
        "tp": int(summary["division_tp"]),
        "fp": int(summary["division_fp"]),
        "fn": int(summary["division_fn"]),
    },
    "per_dataset": compare_rows,
    "gap2_stats": post_stats_by_dataset,
    "decision": decision,
}
(out / "comparison_vs_control.json").write_text(
    json.dumps(comparison, indent=2, sort_keys=True) + "\n"
)
(out / "DONE").write_text("ok\n", encoding="utf-8")

print(f"score:      {float(summary['score']):.12f}")
print(f"control:    {control_score:.12f}")
print(f"delta:      {delta:+.12f}")
print(
    "edges tp/fp/fn:",
    f"{summary['edge_tp']} / {summary['edge_fp']} / {summary['edge_fn']}",
)
print(
    "div tp/fp/fn:",
    f"{summary['division_tp']} / {summary['division_fp']} / {summary['division_fn']}",
)
print(
    f"datasets improved/unchanged/worsened: {n_improved}/{n_unchanged}/{n_worsened}"
)
print(f"gap2 pairs: {summary['gap2_pairs_total']} edges_added: {summary['gap2_added_edges_total']}")
print(f"VERDICT {decision['verdict']} promote={promote} reason={reject_reason}")
print("OUT", out)
PY

echo GAP2_ABLATION_DONE
