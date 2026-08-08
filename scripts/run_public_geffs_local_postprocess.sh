#!/usr/bin/env bash
# Phase 6: public-recipe raw GEFFs + local recipe-C postprocess (det960 yaml postprocess).
# Isolates ensemble/inference vs postprocess as the cause of the public fixed-8 gap.
set -euo pipefail

SRC="${1:-/home/mc46451/biohub-cell-tracking}"
LOG_DIR="${SRC}/logs/slurm"
mkdir -p "${LOG_DIR}"
LOG="${LOG_DIR}/public_geffs_local_postprocess.log"

FIXED8=(
  44b6_0113de3b 44b6_0b24845f 44b6_341df25f 44b6_e57ff5c6
  6bba_05b6850b 6bba_05db0fb1 6bba_969618f6 6bba_fc83837d
)

cd "${SRC}"
STAGE_INPUTS=()
for d in "${FIXED8[@]}"; do
  STAGE_INPUTS+=("data/competition/train/${d}.zarr" "data/competition/train/${d}.geff")
done

echo "Starting public-GEFF local-postprocess at $(date -Iseconds)" | tee "${LOG}"

tar \
  --exclude='.venv' --exclude='.git' --exclude='**/__pycache__' \
  --exclude='outputs' --exclude='logs' \
  --exclude='data/support/repo/predictions' --exclude='data/tmp_seed314159' \
  -cf - \
  README.md LICENSE pyproject.toml configs src scripts \
  data/support \
  "${STAGE_INPUTS[@]}" \
| srun -p gpu_batch -N1 -n1 --gres=gpu:1 --cpus-per-task=8 --mem=64G -t 0-01:30:00 \
  --chdir=/tmp \
  bash -lc "
set -euo pipefail
DEST=/tmp/\${USER}/biohub-public-geffs-local-pp
export OUT_NFS=\${HOME}/biohub-outputs/fixed8/public_geffs_local_postprocess
export RAW_NFS=\${HOME}/biohub-outputs/fixed8/public_two_seed_exact/raw_geff
rm -rf \"\$DEST\"
mkdir -p \"\$DEST\" \"\$OUT_NFS\"
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
test -d \"\$RAW_NFS\"

python - <<'PY'
from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path

import tracksdata as td

from biohub_pipeline import postprocessing
from biohub_pipeline.config import load_config
from biohub_pipeline.fixed8_cv import (
    evaluate_postprocessed_predictions,
    write_metric_outputs,
)
from biohub_pipeline.submission import graph_rows, validate_submission_file, write_rows

FIXED8 = [
    \"44b6_0113de3b\", \"44b6_0b24845f\", \"44b6_341df25f\", \"44b6_e57ff5c6\",
    \"6bba_05b6850b\", \"6bba_05db0fb1\", \"6bba_969618f6\", \"6bba_fc83837d\",
]
out_nfs = Path(os.environ[\"OUT_NFS\"])
raw_nfs = Path(os.environ[\"RAW_NFS\"])
# Local recipe-C postprocess control (det 0.960 yaml); inference fields unused
cfg_path = Path(\"configs/sweeps/two_seed_det_thresh_0_960.yaml\")
config = load_config(cfg_path)
assert float(config.inference[\"ensemble_alpha\"]) == 0.5
assert config.postprocessing[\"output_safe_divisions\"] is True
assert config.postprocessing[\"use_deepcenter_veto\"] is False
assert config.postprocessing[\"output_gap2_recovery\"] is False

out = Path(\"outputs/fixed8/public_geffs_local_postprocess\")
if out.exists():
    shutil.rmtree(out)
out.mkdir(parents=True)
(out / \"config\").mkdir()
shutil.copy2(cfg_path, out / \"config\" / cfg_path.name)
raw_dst = out / \"raw_geff_source\"
raw_dst.mkdir()
geffs = []
for ds in FIXED8:
    src = raw_nfs / f\"{ds}.geff\"
    dest = raw_dst / src.name
    if src.is_dir():
        shutil.copytree(src, dest)
    else:
        shutil.copy2(src, dest)
    geffs.append(dest)

data_dir = Path(\"data/competition/train\")
postprocessing.configure(config.postprocessing, data_dir)
started = time.perf_counter()
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
rows, aggregate = evaluate_postprocessed_predictions(pred_csv, data_dir, config)
summary = {
    **aggregate,
    \"recipe\": \"public_geffs_local_postprocess\",
    \"runtime_seconds\": time.perf_counter() - started,
    \"public_exact_score\": 0.877744529257,
    \"control_det960_score\": 0.887978610299,
    \"delta_vs_public_exact\": float(aggregate[\"score\"]) - 0.877744529257,
    \"delta_vs_det960\": float(aggregate[\"score\"]) - 0.887978610299,
}
manifest = {
    \"schema_version\": 1,
    \"experiment\": \"public_geffs_local_postprocess\",
    \"note\": \"Public dual-seed GEFFs + local recipe-C postprocess (no DeepCenter)\",
    \"raw_geff_source\": str(raw_nfs),
    \"config\": str(cfg_path),
}
write_metric_outputs(out, rows, summary, manifest)
(out / \"DONE\").write_text(\"ok\\n\")
if out_nfs.exists():
    shutil.rmtree(out_nfs)
shutil.copytree(out, out_nfs)
print(\"score\", float(summary[\"score\"]))
print(\"edges\", summary[\"edge_tp\"], summary[\"edge_fp\"], summary[\"edge_fn\"])
print(\"divs\", summary[\"division_tp\"], summary[\"division_fp\"], summary[\"division_fn\"])
print(\"delta_vs_public\", summary[\"delta_vs_public_exact\"])
print(\"delta_vs_det960\", summary[\"delta_vs_det960\"])
print(\"PHASE6_DONE\", out_nfs)
PY
" 2>&1 | tee -a "${LOG}"
