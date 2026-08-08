#!/usr/bin/env bash
# Holdout-8 generalization: local recipe C (α=0.5) at det=0.960 vs 0.96875,
# then safe-div ON vs OFF on the det=0.960 GEFFs.
# Uses double-quoted bash -lc so Python string literals are safe.
set -euo pipefail

SRC="${1:-/home/mc46451/biohub-cell-tracking}"
LOG_DIR="${SRC}/logs/slurm"
mkdir -p "${LOG_DIR}"
LOG="${LOG_DIR}/holdout8_thresh_safediv.log"

HOLDOUT8=(
  44b6_0c582fdc 44b6_0db75fae 44b6_12dfb391 44b6_144b256d
  6bba_062c8d37 6bba_07477033 6bba_07e24132 6bba_085bf656
)

cd "${SRC}"
STAGE_INPUTS=()
for d in "${HOLDOUT8[@]}"; do
  STAGE_INPUTS+=("data/competition/train/${d}.zarr" "data/competition/train/${d}.geff")
done

echo "Starting holdout-8 at $(date -Iseconds)" | tee "${LOG}"

tar \
  --exclude='.venv' --exclude='.git' --exclude='**/__pycache__' \
  --exclude='outputs' --exclude='logs' \
  --exclude='data/support/repo/predictions' --exclude='data/tmp_seed314159' \
  -cf - \
  README.md LICENSE pyproject.toml configs src scripts \
  data/support \
  "${STAGE_INPUTS[@]}" \
| srun -p gpu_batch -N1 -n1 --gres=gpu:1 --cpus-per-task=8 --mem=64G -t 0-06:00:00 \
  --chdir=/tmp \
  bash -lc "
set -euo pipefail
DEST=/tmp/\${USER}/biohub-holdout8
export OUT_ROOT=\${HOME}/biohub-outputs/holdout8
rm -rf \"\$DEST\"
mkdir -p \"\$DEST\" \"\$OUT_ROOT\"
tar xf - -C \"\$DEST\"
cd \"\$DEST\"

source \"\${HOME}/miniconda3/etc/profile.d/conda.sh\"
conda activate biohub
python -m pip install -U pip >/dev/null
python -m pip install --no-deps data/support/wheels/*.whl >/dev/null 2>&1 || true
python -m pip install --find-links=data/support/wheels --prefer-binary \
  tracksdata zarr \"geff>=1.1.3.1.1\" \"geff-spec<1.2\" \"ilpy>=0.5.1\" \
  pyscipopt polars blosc2 dask imagecodecs pyarrow \"rustworkx>=0.17.1\" \
  \"sqlalchemy>=2\" \"scikit-image>=0.24\" \"numcodecs>=0.13,<0.16\" donfig >/dev/null
export PYTHONPATH=\"\${DEST}/src\${PYTHONPATH:+:\$PYTHONPATH}\"

python - <<'PY'
from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path

import pandas as pd
import tracksdata as td

from biohub_pipeline import postprocessing
from biohub_pipeline.config import load_config
from biohub_pipeline.evaluation import official_spec_summarise
from biohub_pipeline.fixed8_cv import (
    evaluate_tables,
    find_fixed8_prediction_geffs,
    _copy_raw_predictions,
    _estimated_total_nodes,
    _read_graph_tables,
    write_metric_outputs,
)
from biohub_pipeline.inference import apply_spatial_d4_patch, build_predict_command, run_prediction
from biohub_pipeline.submission import graph_rows, validate_submission_file, write_rows

HOLDOUT8 = [
    \"44b6_0c582fdc\", \"44b6_0db75fae\", \"44b6_12dfb391\", \"44b6_144b256d\",
    \"6bba_062c8d37\", \"6bba_07477033\", \"6bba_07e24132\", \"6bba_085bf656\",
]
OUT_ROOT = Path(os.environ[\"OUT_ROOT\"])
data_dir = Path(\"data/competition/train\")
repo = Path(\"data/support/repo\")
weights = Path(\"data/support/weights/unet_transformer/split_0/edge_predictor_best.pth\")


def evaluate_datasets(pred_csv: Path, datasets: list[str], config):
    predictions = pd.read_csv(pred_csv)
    found = sorted(predictions[\"dataset\"].unique().tolist())
    if found != sorted(datasets):
        raise RuntimeError(f\"predictions datasets {found} != expected {sorted(datasets)}\")
    scale = tuple(float(v) for v in config.postprocessing[\"voxel_scale_um\"])
    max_distance_um = float(config.local_cv[\"max_match_um\"])
    rows = []
    for dataset in datasets:
        part = predictions[predictions[\"dataset\"] == dataset]
        pred_nodes = part[part[\"row_type\"] == \"node\"].loc[:, [\"node_id\", \"t\", \"z\", \"y\", \"x\"]]
        pred_edges = part[part[\"row_type\"] == \"edge\"].loc[:, [\"source_id\", \"target_id\"]]
        gt_path = data_dir / f\"{dataset}.geff\"
        gt_nodes, gt_edges = _read_graph_tables(gt_path)
        rows.append(
            evaluate_tables(
                dataset, pred_nodes, pred_edges, gt_nodes, gt_edges,
                _estimated_total_nodes(gt_path), scale=scale, max_distance_um=max_distance_um,
            )
        )
    summary = {**official_spec_summarise(rows), \"datasets\": list(datasets)}
    return rows, summary


def convert_geffs(geffs, config_path: Path, out: Path, label: str):
    config = load_config(config_path)
    postprocessing.configure(config.postprocessing, data_dir)
    pred_csv = out / \"predictions\" / \"postprocessed_submission.csv\"
    all_rows = []
    for geff_path in geffs:
        dataset = geff_path.stem
        loaded = td.graph.IndexedRXGraph.from_geff(geff_path)
        graph = loaded[0] if isinstance(loaded, tuple) else loaded
        nodes = {
            int(r[\"node_id\"]): {
                \"node_id\": int(r[\"node_id\"]), \"t\": int(r[\"t\"]),
                \"z\": float(r[\"z\"]), \"y\": float(r[\"y\"]), \"x\": float(r[\"x\"]),
            }
            for r in graph.node_attrs().iter_rows(named=True)
        }
        edges = []
        for r in graph.edge_attrs().iter_rows(named=True):
            p = r.get(\"edge_prob\") if hasattr(r, \"get\") else None
            edges.append({
                \"source_id\": int(r[\"source_id\"]),
                \"target_id\": int(r[\"target_id\"]),
                \"edge_prob\": None if p is None else float(p),
            })
        nodes, edges, _stats = postprocessing.filter_output_graph(nodes, edges, dataset=dataset)
        all_rows.extend(graph_rows(dataset, nodes, edges, len(all_rows)))
    write_rows(all_rows, pred_csv)
    validate_submission_file(pred_csv)
    rows, aggregate = evaluate_datasets(pred_csv, HOLDOUT8, config)
    summary = {
        **aggregate,
        \"label\": label,
        \"detection_threshold\": float(config.inference[\"detection_threshold\"]),
        \"output_safe_divisions\": bool(config.postprocessing[\"output_safe_divisions\"]),
        \"ensemble_alpha\": float(config.inference.get(\"ensemble_alpha\", 0.5)),
    }
    manifest = {
        \"schema_version\": 1,
        \"experiment\": label,
        \"datasets\": HOLDOUT8,
        \"config\": str(config_path),
        \"note\": \"Holdout-8 unused for selecting det=0.960; local recipe C alpha=0.5\",
    }
    write_metric_outputs(out, rows, summary, manifest)
    (out / \"DONE\").write_text(\"ok\\n\")
    return rows, summary


def run_infer(det: float, out_name: str):
    if abs(det - 0.960) < 1e-12:
        config_path = Path(\"configs/experiments/holdout_det0_960_safeon.yaml\")
    elif abs(det - 0.96875) < 1e-12:
        config_path = Path(\"configs/experiments/holdout_det0_96875_safeon.yaml\")
    else:
        raise ValueError(det)
    config = load_config(config_path)
    assert abs(float(config.inference[\"detection_threshold\"]) - det) < 1e-12
    assert float(config.inference[\"ensemble_alpha\"]) == 0.5
    assert config.postprocessing[\"output_safe_divisions\"] is True
    assert config.postprocessing[\"use_deepcenter_veto\"] is False

    out = OUT_ROOT / out_name
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    (out / \"config\").mkdir()
    shutil.copy2(config_path, out / \"config\" / config_path.name)

    pred_root = repo / \"predictions\"
    if pred_root.exists():
        shutil.rmtree(pred_root)

    if config.inference[\"spatial_d4_tta\"]:
        apply_spatial_d4_patch(repo, str(config.inference[\"prediction_script\"]))
    cmd, _ = build_predict_command(config, data_dir.resolve(), repo, weights.resolve(), HOLDOUT8)
    print(\"INFER\", out_name, \"CMD\", \" \".join(cmd), flush=True)
    t0 = time.perf_counter()
    run_prediction(cmd, repo)
    geffs = find_fixed8_prediction_geffs(repo, \"unet_transformer\", HOLDOUT8)
    _copy_raw_predictions(geffs, out, config_path=config_path)
    rows, summary = convert_geffs(geffs, config_path, out, out_name)
    summary[\"runtime_seconds\"] = time.perf_counter() - t0
    (out / \"summary.json\").write_text(json.dumps(summary, indent=2, sort_keys=True) + \"\\n\")
    print(
        \"%s: score=%.12f edge=%s/%s/%s div=%s/%s/%s\"
        % (
            out_name,
            float(summary[\"score\"]),
            summary[\"edge_tp\"], summary[\"edge_fp\"], summary[\"edge_fn\"],
            summary[\"division_tp\"], summary[\"division_fp\"], summary[\"division_fn\"],
        ),
        flush=True,
    )
    return geffs


run_infer(0.960, \"det0_960_safeon\")
run_infer(0.96875, \"det0_96875_safeon\")

cfg_off = Path(\"configs/experiments/holdout_det0_960_safeoff.yaml\")
out_off = OUT_ROOT / \"det0_960_safeoff\"
if out_off.exists():
    shutil.rmtree(out_off)
out_off.mkdir(parents=True)
(out_off / \"config\").mkdir()
shutil.copy2(cfg_off, out_off / \"config\" / cfg_off.name)

raw_dir = OUT_ROOT / \"det0_960_safeon\" / \"raw_geff\"
if not raw_dir.is_dir():
    raw_dir = OUT_ROOT / \"det0_960_safeon\" / \"predictions\" / \"raw_geff\"
geffs = [raw_dir / f\"{d}.geff\" for d in HOLDOUT8]
assert all(p.exists() for p in geffs), geffs
src_geff = out_off / \"raw_geff_source\"
src_geff.mkdir()
for p in geffs:
    if p.is_dir():
        shutil.copytree(p, src_geff / p.name)
    else:
        shutil.copy2(p, src_geff / p.name)

rows_off, summary_off = convert_geffs(geffs, cfg_off, out_off, \"det0_960_safeoff\")
print(
    \"det0_960_safeoff: score=%.12f edge=%s/%s/%s div=%s/%s/%s\"
    % (
        float(summary_off[\"score\"]),
        summary_off[\"edge_tp\"], summary_off[\"edge_fp\"], summary_off[\"edge_fn\"],
        summary_off[\"division_tp\"], summary_off[\"division_fp\"], summary_off[\"division_fn\"],
    ),
    flush=True,
)

s960 = json.loads((OUT_ROOT / \"det0_960_safeon\" / \"summary.json\").read_text())
s968 = json.loads((OUT_ROOT / \"det0_96875_safeon\" / \"summary.json\").read_text())
soff = summary_off

def per_ds(path):
    return pd.read_csv(path).set_index(\"dataset\")

p960 = per_ds(OUT_ROOT / \"det0_960_safeon\" / \"per_dataset.csv\")
p968 = per_ds(OUT_ROOT / \"det0_96875_safeon\" / \"per_dataset.csv\")
poff = per_ds(out_off / \"per_dataset.csv\")

cmp = {
    \"holdout_datasets\": HOLDOUT8,
    \"det960_safeon\": {
        \"score\": float(s960[\"score\"]),
        \"edge_tp_fp_fn\": [s960[\"edge_tp\"], s960[\"edge_fp\"], s960[\"edge_fn\"]],
        \"div_tp_fp_fn\": [s960[\"division_tp\"], s960[\"division_fp\"], s960[\"division_fn\"]],
    },
    \"det96875_safeon\": {
        \"score\": float(s968[\"score\"]),
        \"edge_tp_fp_fn\": [s968[\"edge_tp\"], s968[\"edge_fp\"], s968[\"edge_fn\"]],
        \"div_tp_fp_fn\": [s968[\"division_tp\"], s968[\"division_fp\"], s968[\"division_fn\"]],
    },
    \"det960_safeoff\": {
        \"score\": float(soff[\"score\"]),
        \"edge_tp_fp_fn\": [soff[\"edge_tp\"], soff[\"edge_fp\"], soff[\"edge_fn\"]],
        \"div_tp_fp_fn\": [soff[\"division_tp\"], soff[\"division_fp\"], soff[\"division_fn\"]],
    },
    \"delta_960_minus_96875\": float(s960[\"score\"]) - float(s968[\"score\"]),
    \"delta_safeoff_minus_safeon_960\": float(soff[\"score\"]) - float(s960[\"score\"]),
    \"per_dataset_960_vs_96875\": {},
    \"per_dataset_safeoff_vs_safeon\": {},
}
wins_960 = 0
wins_968 = 0
for ds in HOLDOUT8:
    d = float(p960.loc[ds, \"adj_edge_jaccard\"]) - float(p968.loc[ds, \"adj_edge_jaccard\"])
    cmp[\"per_dataset_960_vs_96875\"][ds] = d
    if d > 1e-12:
        wins_960 += 1
    elif d < -1e-12:
        wins_968 += 1
for ds in HOLDOUT8:
    d = float(poff.loc[ds, \"adj_edge_jaccard\"]) - float(p960.loc[ds, \"adj_edge_jaccard\"])
    cmp[\"per_dataset_safeoff_vs_safeon\"][ds] = d
cmp[\"datasets_won_by_960\"] = wins_960
cmp[\"datasets_won_by_96875\"] = wins_968
(OUT_ROOT / \"comparison.json\").write_text(json.dumps(cmp, indent=2, sort_keys=True) + \"\\n\")
print(\"COMPARISON\", json.dumps(cmp, indent=2, sort_keys=True), flush=True)
(OUT_ROOT / \"DONE\").write_text(\"ok\\n\")
print(\"HOLDOUT8_DONE\", OUT_ROOT, flush=True)
PY
" 2>&1 | tee -a "${LOG}"

if srun -p gpu_batch -N1 -n1 --gres=gpu:1 -t 00:05:00 --mem=2G --chdir=/tmp \
  bash -lc 'test -f ${HOME}/biohub-outputs/holdout8/comparison.json' 2>/dev/null; then
  mkdir -p "${SRC}/outputs/holdout8"
  srun -p gpu_batch -N1 -n1 --gres=gpu:1 -t 00:10:00 --mem=4G --chdir=/tmp bash -lc '
    cd ${HOME}/biohub-outputs/holdout8 && tar cf - comparison.json DONE \
      det0_960_safeon/summary.json det0_960_safeon/per_dataset.csv \
      det0_96875_safeon/summary.json det0_96875_safeon/per_dataset.csv \
      det0_960_safeoff/summary.json det0_960_safeoff/per_dataset.csv
  ' | tar xf - -C "${SRC}/outputs/holdout8"
  echo "LOGIN_HOLDOUT_MIRROR_OK"
fi
