#!/usr/bin/env bash
# Attribute bottleneck postprocessing_removed edges to the first postprocess stage
# that drops them. Postprocess-only; reuses saved raw GEFFs + diagnostic CSVs.
set -euo pipefail

SRC="${1:-/home/mc46451/biohub-cell-tracking}"
LOG_DIR="${SRC}/logs/slurm"
mkdir -p "${LOG_DIR}"
LOG="${LOG_DIR}/postprocess_stage_attribution.log"

FIXED8=(
  44b6_0113de3b 44b6_0b24845f 44b6_341df25f 44b6_e57ff5c6
  6bba_05b6850b 6bba_05db0fb1 6bba_969618f6 6bba_fc83837d
)
HOLDOUT8=(
  44b6_0c582fdc 44b6_0db75fae 44b6_12dfb391 44b6_144b256d
  6bba_062c8d37 6bba_07477033 6bba_07e24132 6bba_085bf656
)

cd "${SRC}"
STAGE_INPUTS=()
for d in "${FIXED8[@]}" "${HOLDOUT8[@]}"; do
  STAGE_INPUTS+=("data/competition/train/${d}.zarr" "data/competition/train/${d}.geff")
done

# Stage diagnostics outside excluded outputs/ tree so tar includes them.
DIAG_STAGE="${SRC}/.tmp_stage_attr_diags"
rm -rf "${DIAG_STAGE}"
mkdir -p "${DIAG_STAGE}"
cp -f \
  "${SRC}/outputs/experiments/candidate_edge_bottleneck_v1/candidate_edge_diagnostic.csv" \
  "${DIAG_STAGE}/fixed8_candidate_edge_diagnostic.csv"
cp -f \
  "${SRC}/outputs/experiments/candidate_edge_bottleneck_holdout8/candidate_edge_diagnostic.csv" \
  "${DIAG_STAGE}/holdout8_candidate_edge_diagnostic.csv"

echo "Starting postprocess stage attribution at $(date -Iseconds)" | tee "${LOG}"

tar \
  --exclude='.venv' --exclude='.git' --exclude='**/__pycache__' \
  --exclude='outputs' --exclude='logs' \
  --exclude='data/support/repo/predictions' --exclude='data/tmp_seed314159' \
  -cf - \
  README.md LICENSE pyproject.toml configs src scripts \
  data/support \
  .tmp_stage_attr_diags \
  "${STAGE_INPUTS[@]}" \
| srun -p gpu_batch -N1 -n1 --gres=gpu:1 --cpus-per-task=8 --mem=32G -t 0-00:45:00 \
  --chdir=/tmp \
  bash -lc "
set -euo pipefail
DEST=/tmp/\${USER}/biohub-pp-stage-attr
OUT_NFS=\${HOME}/biohub-outputs/experiments/postprocess_stage_attribution_v1
FIXED_RAW=\${HOME}/biohub-outputs/experiments/candidate_edge_bottleneck_v1/raw_geff
HOLD_RAW=\${HOME}/biohub-outputs/holdout8/det0_96875_safeon/raw_geff
if [[ ! -d \"\$HOLD_RAW\" ]]; then
  HOLD_RAW=\${HOME}/biohub-outputs/experiments/candidate_edge_bottleneck_holdout8/raw_geff
fi

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

test -d \"\$FIXED_RAW\"
test -d \"\$HOLD_RAW\"
mkdir -p \"\$OUT_NFS\"

export FIXED_RAW HOLD_RAW OUT_NFS
python - <<'PY'
from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path

import pandas as pd
import tracksdata as td

from biohub_pipeline import postprocessing
from biohub_pipeline.config import load_config
from biohub_pipeline.postprocess_stage_trace import filter_output_graph_traced

FIXED_RAW = Path(os.environ['FIXED_RAW'])
HOLD_RAW = Path(os.environ['HOLD_RAW'])
OUT_NFS = Path(os.environ['OUT_NFS'])
DATA_DIR = Path('data/competition/train')
CFG = Path('configs/sweeps/two_seed_det_thresh_0_96875.yaml')
if not CFG.exists():
    CFG = Path('configs/clean_v106_two_seed.yaml')

FIXED_DIAG = Path('.tmp_stage_attr_diags/fixed8_candidate_edge_diagnostic.csv')
HOLD_DIAG = Path('.tmp_stage_attr_diags/holdout8_candidate_edge_diagnostic.csv')

config = load_config(CFG)
assert abs(float(config.inference['detection_threshold']) - 0.96875) < 1e-12
assert config.postprocessing['output_safe_divisions'] is True
assert config.postprocessing['output_filter_short_tracks'] is True
postprocessing.configure(config.postprocessing, DATA_DIR)


def load_raw(geff_path: Path):
    loaded = td.graph.IndexedRXGraph.from_geff(geff_path)
    graph = loaded[0] if isinstance(loaded, tuple) else loaded
    nodes = {
        int(r['node_id']): {
            'node_id': int(r['node_id']), 't': int(r['t']),
            'z': float(r['z']), 'y': float(r['y']), 'x': float(r['x']),
        }
        for r in graph.node_attrs().iter_rows(named=True)
    }
    edges = []
    for r in graph.edge_attrs().iter_rows(named=True):
        p = r.get('edge_prob') if hasattr(r, 'get') else None
        edges.append({
            'source_id': int(r['source_id']),
            'target_id': int(r['target_id']),
            'edge_prob': None if p is None else float(p),
        })
    return nodes, edges


def attribute_set(name: str, raw_dir: Path, diag_path: Path) -> dict:
    diag = pd.read_csv(diag_path)
    pp = diag[diag['cause'] == 'postprocessing_removed'].copy()
    rows = []
    stage_counts = Counter()
    snapshots_by_ds = {}
    for dataset, part in pp.groupby('dataset'):
        watch = {
            (int(r.pred_source_id), int(r.pred_target_id))
            for r in part.itertuples(index=False)
            if pd.notna(r.pred_source_id) and pd.notna(r.pred_target_id)
        }
        geff = raw_dir / f'{dataset}.geff'
        assert geff.exists(), geff
        nodes, edges = load_raw(geff)
        raw_set = {(int(e['source_id']), int(e['target_id'])) for e in edges}
        _, final_edges, stats, trace = filter_output_graph_traced(
            nodes, edges, dataset=dataset, watch=watch
        )
        final_set = {(int(e['source_id']), int(e['target_id'])) for e in final_edges}
        snapshots_by_ds[dataset] = trace['stage_snapshots']
        first_loss = trace['first_loss_by_edge']
        for r in part.itertuples(index=False):
            if pd.isna(r.pred_source_id) or pd.isna(r.pred_target_id):
                stage = 'missing_pred_ids'
                in_raw = False
                in_final = False
                ps = pt = None
            else:
                ps, pt = int(r.pred_source_id), int(r.pred_target_id)
                edge = (ps, pt)
                in_raw = edge in raw_set
                in_final = edge in final_set
                if not in_raw:
                    stage = 'not_in_raw_geff'
                else:
                    stage = first_loss.get(edge, 'unknown')
            stage_counts[stage] += 1
            rows.append({
                'split': name,
                'dataset': dataset,
                'gt_source_id': int(r.gt_source_id),
                'gt_target_id': int(r.gt_target_id),
                'pred_source_id': ps,
                'pred_target_id': pt,
                'source_in_final_diagnostic': bool(r.source_in_final),
                'target_in_final_diagnostic': bool(r.target_in_final),
                'in_raw_geff': in_raw,
                'in_final_traced': in_final,
                'first_loss_stage': stage,
            })
        print(name, dataset, 'watched', len(watch), 'stage_counts_partial', dict(Counter(first_loss.values())), flush=True)

    frame = pd.DataFrame(rows)
    out = OUT_NFS / name
    out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out / 'edge_stage_attribution.csv', index=False)
    summary = {
        'split': name,
        'n_postprocessing_removed': int(len(frame)),
        'stage_counts': dict(stage_counts),
        'stage_fractions': {
            k: float(v) / len(frame) if len(frame) else 0.0 for k, v in stage_counts.items()
        },
        'config': str(CFG),
        'raw_geff': str(raw_dir),
        'diagnostic': str(diag_path),
        'motion_relink': bool(config.postprocessing['output_motion_relink']),
        'single_parent_repair': bool(config.postprocessing['output_single_parent_repair']),
        'single_child_repair': bool(config.postprocessing['output_single_child_repair']),
        'short_track_filter': bool(config.postprocessing['output_filter_short_tracks']),
        'safe_divisions': bool(config.postprocessing['output_safe_divisions']),
    }
    (out / 'summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n')
    (out / 'stage_snapshots.json').write_text(json.dumps(snapshots_by_ds, indent=2) + '\n')
    return summary


fixed = attribute_set('fixed8', FIXED_RAW, FIXED_DIAG)
hold = attribute_set('holdout8', HOLD_RAW, HOLD_DIAG)
combined = {
    'fixed8': fixed,
    'holdout8': hold,
    'dominant_fixed8': max(fixed['stage_counts'], key=fixed['stage_counts'].get) if fixed['stage_counts'] else None,
    'dominant_holdout8': max(hold['stage_counts'], key=hold['stage_counts'].get) if hold['stage_counts'] else None,
}
(OUT_NFS / 'combined_summary.json').write_text(json.dumps(combined, indent=2, sort_keys=True) + '\n')
(OUT_NFS / 'DONE').write_text('ok\n')
print(json.dumps(combined, indent=2, sort_keys=True))
PY
echo DONE_ATTRIBUTION
" 2>&1 | tee -a "${LOG}"

rm -rf "${DIAG_STAGE}"
mkdir -p "${SRC}/outputs/experiments/postprocess_stage_attribution_v1"
srun -p gpu_batch -N1 -n1 --gres=gpu:1 -t 00:08:00 --mem=4G --chdir=/tmp bash -lc '
  cd ${HOME}/biohub-outputs/experiments/postprocess_stage_attribution_v1 && \
  tar cf - DONE combined_summary.json \
    fixed8/summary.json fixed8/edge_stage_attribution.csv fixed8/stage_snapshots.json \
    holdout8/summary.json holdout8/edge_stage_attribution.csv holdout8/stage_snapshots.json
' | tar xf - -C "${SRC}/outputs/experiments/postprocess_stage_attribution_v1"
echo "Attribution finished at $(date -Iseconds)" | tee -a "${LOG}"
cat "${SRC}/outputs/experiments/postprocess_stage_attribution_v1/combined_summary.json"
