#!/usr/bin/env bash
# Re-run public-recipe postprocess+eval from saved raw GEFFs (no re-inference).
set -euo pipefail

SRC="${1:-/home/mc46451/biohub-cell-tracking}"
LOG_DIR="${SRC}/logs/slurm"
mkdir -p "${LOG_DIR}"
LOG="${LOG_DIR}/public_two_seed_exact_rescore.log"

FIXED8=(
  44b6_0113de3b 44b6_0b24845f 44b6_341df25f 44b6_e57ff5c6
  6bba_05b6850b 6bba_05db0fb1 6bba_969618f6 6bba_fc83837d
)

cd "${SRC}"
STAGE_INPUTS=()
for d in "${FIXED8[@]}"; do
  STAGE_INPUTS+=("data/competition/train/${d}.zarr" "data/competition/train/${d}.geff")
done

echo "Rescoring public recipe from NFS GEFFs at $(date -Iseconds)" | tee "${LOG}"

tar \
  --exclude='.venv' --exclude='.git' --exclude='**/__pycache__' \
  --exclude='outputs' --exclude='logs' \
  --exclude='data/support/repo/predictions' --exclude='data/tmp_seed314159' \
  -cf - \
  README.md LICENSE pyproject.toml configs src scripts \
  data/support data/deepcenter \
  "${STAGE_INPUTS[@]}" \
| srun -p gpu_batch -N1 -n1 --gres=gpu:1 --cpus-per-task=8 --mem=64G -t 0-02:00:00 \
  --chdir=/tmp \
  bash -lc "
set -euo pipefail
DEST=/tmp/\${USER}/biohub-public-recipe-rescore
export OUT_NFS=\${HOME}/biohub-outputs/fixed8/public_two_seed_exact
export RAW_NFS=\${HOME}/biohub-outputs/fixed8/public_two_seed_exact/raw_geff
# If previous failed before NFS copy, raw may only be on node /tmp from prior job — expect NFS or fail.
rm -rf \"\$DEST\"
mkdir -p \"\$DEST\"
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

# Prefer GEFFs already on NFS; else fail clearly
if [[ ! -d \"\$RAW_NFS\" ]]; then
  # try leftover local stage from failed job on this node
  ALT=/tmp/\${USER}/biohub-public-recipe-fixed8/outputs/fixed8/public_two_seed_exact/raw_geff
  if [[ -d \"\$ALT\" ]]; then
    mkdir -p \"\$OUT_NFS\"
    cp -a \"\$ALT\" \"\$OUT_NFS/\"
    RAW_NFS=\"\$OUT_NFS/raw_geff\"
  else
    echo \"ERROR: no raw_geff at \$RAW_NFS or \$ALT\" >&2
    ls -la \"\${HOME}/biohub-outputs/fixed8/public_two_seed_exact\" || true
    exit 2
  fi
fi

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
    CURRENT_V106_REFERENCE_SCORE,
    MATERIAL_DELTA,
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
cfg_path = Path(\"configs/experiments/public_two_seed_exact.yaml\")
config = load_config(cfg_path)
data_dir = Path(\"data/competition/train\")

# Work in a fresh local out, then replace NFS atomically-ish
out = Path(\"outputs/fixed8/public_two_seed_exact\")
if out.exists():
    shutil.rmtree(out)
out.mkdir(parents=True)
(out / \"config\").mkdir()
shutil.copy2(cfg_path, out / \"config\" / cfg_path.name)
raw_dst = out / \"raw_geff\"
raw_dst.mkdir(parents=True)
geffs = []
for ds in FIXED8:
    src = raw_nfs / f\"{ds}.geff\"
    assert src.exists(), src
    dest = raw_dst / src.name
    if src.is_dir():
        shutil.copytree(src, dest)
    else:
        shutil.copy2(src, dest)
    geffs.append(dest)
# also legacy path
legacy = out / \"predictions\" / \"raw_geff\"
legacy.mkdir(parents=True)
for g in geffs:
    d = legacy / g.name
    if g.is_dir():
        shutil.copytree(g, d)
    else:
        shutil.copy2(g, d)

postprocessing.configure(config.postprocessing, data_dir)
bundle = postprocessing.load_deepcenter_veto_detector()
assert bundle is not None

started = time.perf_counter()
pred_csv = out / \"predictions\" / \"postprocessed_submission.csv\"
all_rows = []
stats_by_ds = {}
for geff_path in geffs:
    dataset = geff_path.stem
    loaded = td.graph.IndexedRXGraph.from_geff(geff_path)
    graph = loaded[0] if isinstance(loaded, tuple) else loaded
    nodes = {
        int(r[\"node_id\"]): {
            \"node_id\": int(r[\"node_id\"]),
            \"t\": int(r[\"t\"]),
            \"z\": float(r[\"z\"]),
            \"y\": float(r[\"y\"]),
            \"x\": float(r[\"x\"]),
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
    nodes, edges, stats = postprocessing.filter_output_graph(
        nodes, edges, dataset=dataset, deepcenter_bundle=bundle
    )
    stats_by_ds[dataset] = {k: int(v) if isinstance(v, (int, float)) else v for k, v in stats.items()}
    all_rows.extend(graph_rows(dataset, nodes, edges, len(all_rows)))

write_rows(all_rows, pred_csv)
validate_submission_file(pred_csv)
(out / \"predictions\" / \"postprocess_stats.json\").write_text(
    json.dumps(stats_by_ds, indent=2, sort_keys=True) + \"\\n\"
)
rows, aggregate = evaluate_postprocessed_predictions(pred_csv, data_dir, config)
runtime = time.perf_counter() - started
summary = {
    **aggregate,
    \"detection_threshold\": 0.96875,
    \"recipe\": \"public_two_seed_exact\",
    \"secondary_detection_weight\": 0.475,
    \"secondary_edge_weight\": 0.15,
    \"secondary_link_mode\": \"low_margin_consensus\",
    \"dual_seed_edge_threshold\": 0.48,
    \"ilp_disappearance_weight\": 1.5,
    \"runtime_seconds\": runtime,
    \"control_det960_score\": 0.887978610299,
    \"control_safeoff_score\": 0.888330075629,
    \"delta_vs_det960\": float(aggregate[\"score\"]) - 0.887978610299,
    \"delta_vs_safeoff\": float(aggregate[\"score\"]) - 0.888330075629,
    \"reference_score\": CURRENT_V106_REFERENCE_SCORE,
    \"delta_vs_reference\": float(aggregate[\"score\"]) - CURRENT_V106_REFERENCE_SCORE,
    \"reference_difference_material\": abs(float(aggregate[\"score\"]) - CURRENT_V106_REFERENCE_SCORE) > MATERIAL_DELTA,
    \"rescored_from_saved_geffs\": True,
}
manifest = {
    \"schema_version\": 1,
    \"experiment\": \"public_two_seed_exact\",
    \"config\": str(cfg_path),
    \"note\": \"Public Pilkwang dual-seed postprocess rescored from saved GEFFs; DeepCenter gap gate on\",
    \"raw_geff_source\": str(raw_nfs),
}
write_metric_outputs(out, rows, summary, manifest)
(out / \"DONE\").write_text(\"ok\\n\")
print(\"score:\", float(summary[\"score\"]))
print(\"edges:\", summary[\"edge_tp\"], summary[\"edge_fp\"], summary[\"edge_fn\"])
print(\"divs:\", summary[\"division_tp\"], summary[\"division_fp\"], summary[\"division_fn\"])
print(\"delta_vs_det960:\", summary[\"delta_vs_det960\"])
print(\"delta_vs_safeoff:\", summary[\"delta_vs_safeoff\"])

# Merge into NFS without deleting raw_geff
out_nfs.mkdir(parents=True, exist_ok=True)
for name in [\"summary.json\", \"per_dataset.csv\", \"manifest.json\", \"DONE\"]:
    shutil.copy2(out / name, out_nfs / name)
shutil.copytree(out / \"predictions\", out_nfs / \"predictions\", dirs_exist_ok=True)
shutil.copytree(out / \"config\", out_nfs / \"config\", dirs_exist_ok=True)
print(\"PUBLIC_RECIPE_RESCORE_DONE\", out_nfs)
PY
" 2>&1 | tee -a "${LOG}"

# Fetch summary to login
mkdir -p "${SRC}/outputs/fixed8/public_two_seed_exact"
srun -p gpu_batch -N1 -n1 --gres=gpu:1 -t 00:05:00 --mem=4G --chdir=/tmp bash -lc '
  cd ${HOME}/biohub-outputs/fixed8/public_two_seed_exact && tar cf - summary.json per_dataset.csv manifest.json DONE predictions/postprocessed_submission.csv predictions/postprocess_stats.json config 2>/dev/null
' | tar xf - -C "${SRC}/outputs/fixed8/public_two_seed_exact"
echo "LOGIN_MIRROR_OK"
cat "${SRC}/outputs/fixed8/public_two_seed_exact/summary.json"
