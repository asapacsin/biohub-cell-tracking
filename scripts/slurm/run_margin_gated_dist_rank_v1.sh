#!/usr/bin/env bash
# GPU eval: margin-gated distance rank under frozen recipe_c edge=0.40 motion OFF.
# Does not change motion_relink / edge_threshold / det / ensemble.
# Control: fixed-8 0.9181439782806684 / holdout-8 0.9646726188580379
set -euo pipefail

SRC="${1:-/home/mc46451/biohub-cell-tracking}"
LOG_DIR="${SRC}/logs/slurm"
mkdir -p "${LOG_DIR}"
LOG="${LOG_DIR}/margin_gated_dist_rank_v1.log"

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

echo "Starting margin_gated_dist_rank_v1 at $(date -Iseconds)" | tee "${LOG}"

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
DEST=/tmp/\${USER}/biohub-margin-gated-dist-v1
OUT_NFS=\${HOME}/biohub-outputs/experiments/margin_gated_dist_rank_v1
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

if [[ -e \"\$OUT_NFS\" ]]; then
  echo \"ERROR: output already exists: \$OUT_NFS\" >&2
  exit 2
fi
mkdir -p \"\$OUT_NFS\"
export OUT_NFS

python - <<'PY'
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

from biohub_pipeline.config import load_config
from biohub_pipeline.evaluation import official_spec_summarise
from biohub_pipeline.fixed8_cv import (
    FIXED8_DATASETS,
    _copy_raw_predictions,
    _estimated_total_nodes,
    _read_graph_tables,
    evaluate_postprocessed_predictions,
    evaluate_tables,
    find_fixed8_prediction_geffs,
    write_metric_outputs,
)
from biohub_pipeline.inference import (
    apply_edge_diagnostic_patch,
    apply_spatial_d4_patch,
    build_predict_command,
    run_prediction,
)
from biohub_pipeline.submission import write_submission_from_geff

DATA_DIR = Path('data/competition/train').resolve()
SUPPORT = Path('data/support').resolve()
CFG = Path('configs/experiments/recipe_c_edge_0_40_margin_gated_dist_v1.yaml').resolve()
OUT = Path('outputs/experiments/margin_gated_dist_rank_v1').resolve()
OUT_NFS = Path(__import__('os').environ['OUT_NFS']).resolve()

CONTROL_FIXED = 0.9181439782806684
CONTROL_HOLD = 0.9646726188580379
HOLDOUT8 = [
    '44b6_0c582fdc', '44b6_0db75fae', '44b6_12dfb391', '44b6_144b256d',
    '6bba_062c8d37', '6bba_07477033', '6bba_07e24132', '6bba_085bf656',
]

config = load_config(CFG)
assert abs(float(config.inference['edge_threshold']) - 0.40) < 1e-12
assert config.postprocessing['output_motion_relink'] is False
assert abs(float(config.inference['detection_threshold']) - 0.96875) < 1e-12
assert abs(float(config.inference['margin_gated_dist_lambda']) - 0.1) < 1e-12
assert abs(float(config.inference['margin_gated_dist_delta']) - 0.15) < 1e-12
assert abs(float(config.inference['margin_gated_dist_dens_min']) - 8.0) < 1e-12
print(
    'RECIPE_OK edge=', config.inference['edge_threshold'],
    'motion=', config.postprocessing['output_motion_relink'],
    'lam=', config.inference['margin_gated_dist_lambda'],
    flush=True,
)


def evaluate_custom(pred_csv, datasets):
    import pandas as pd
    predictions = pd.read_csv(pred_csv)
    scale = tuple(float(v) for v in config.postprocessing['voxel_scale_um'])
    max_distance_um = float(config.local_cv['max_match_um'])
    rows = []
    for dataset in datasets:
        part = predictions[predictions['dataset'] == dataset]
        pred_nodes = part[part['row_type'] == 'node'].loc[:, ['node_id', 't', 'z', 'y', 'x']]
        pred_edges = part[part['row_type'] == 'edge'].loc[:, ['source_id', 'target_id']]
        gt_nodes, gt_edges = _read_graph_tables(DATA_DIR / f'{dataset}.geff')
        rows.append(
            evaluate_tables(
                dataset, pred_nodes, pred_edges, gt_nodes, gt_edges,
                _estimated_total_nodes(DATA_DIR / f'{dataset}.geff'),
                scale=scale, max_distance_um=max_distance_um,
            )
        )
    return rows, {**official_spec_summarise(rows), 'datasets': list(datasets)}


def run_split(name, datasets, control_score):
    out = OUT / name
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    work = out / 'work'
    work.mkdir()
    repo = work / 'support_repo'
    shutil.copytree(SUPPORT / 'repo', repo)
    if config.inference['spatial_d4_tta']:
        apply_spatial_d4_patch(repo, str(config.inference['prediction_script']))
    primary = SUPPORT / Path(str(config.inference['weights_relative']))
    command, _ = build_predict_command(config, DATA_DIR, repo, primary, datasets)
    apply_edge_diagnostic_patch(repo, str(config.inference['prediction_script']))
    capture_dir = out / 'candidate_capture'
    capture_dir.mkdir(parents=True)
    command = list(command) + [
        '--edge-diagnostic-dir', str(capture_dir),
        '--edge-diagnostic-top-k', '16',
    ]
    (out / 'command.txt').write_text(' '.join(command) + '\n')
    print(
        'RUN', name,
        'has_margin_lambda', '--margin-gated-dist-lambda' in command,
        'has_edge_threshold', '--edge-threshold' in command,
        flush=True,
    )
    started = time.perf_counter()
    run_prediction(command, repo)
    geffs = find_fixed8_prediction_geffs(repo, str(config.inference['method']), datasets)
    raw_dir = _copy_raw_predictions(geffs, out, config_path=CFG)
    pred_csv = out / 'predictions' / 'postprocessed_submission.csv'
    write_submission_from_geff(geffs, config, DATA_DIR, pred_csv)
    if datasets == list(FIXED8_DATASETS):
        rows, aggregate = evaluate_postprocessed_predictions(pred_csv, DATA_DIR, config)
    else:
        rows, aggregate = evaluate_custom(pred_csv, datasets)
    runtime = time.perf_counter() - started
    summary = {
        **aggregate,
        'label': f'{name}_margin_gated_dist_rank_v1',
        'edge_threshold': 0.40,
        'output_motion_relink': False,
        'detection_threshold': 0.96875,
        'margin_gated_dist_lambda': 0.1,
        'margin_gated_dist_delta': 0.15,
        'margin_gated_dist_dens_min': 8.0,
        'delta_vs_control': float(aggregate['score']) - control_score,
        'control_score': control_score,
        'runtime_seconds': runtime,
    }
    write_metric_outputs(
        out, rows, summary,
        {
            'schema_version': 1,
            'experiment': 'margin_gated_dist_rank_v1',
            'split': name,
            'config': str(CFG),
            'raw_geff': str(raw_dir),
            'note': 'Frozen recipe + margin-gated distance rank term',
        },
    )
    shutil.copy2(CFG, out / 'experiment_config.yaml')
    (out / 'DONE').write_text('ok\n')
    print(name, 'score', summary['score'], 'delta', summary['delta_vs_control'], flush=True)
    return summary


if OUT.exists():
    shutil.rmtree(OUT)
OUT.mkdir(parents=True)
shutil.copy2(CFG, OUT / 'experiment_config.yaml')
fixed = run_split('fixed8', list(FIXED8_DATASETS), CONTROL_FIXED)
hold = run_split('holdout8', HOLDOUT8, CONTROL_HOLD)
comparison = {
    'hypothesis': (
        'Margin-gated pre-softmax +lam*dist on near-tie dense columns recovers '
        'rank-2 ordinary associations without destroying short true edges'
    ),
    'control_fixed8': CONTROL_FIXED,
    'control_holdout8': CONTROL_HOLD,
    'params': {
        'margin_gated_dist_lambda': 0.1,
        'margin_gated_dist_delta': 0.15,
        'margin_gated_dist_dens_min': 8.0,
        'margin_gated_dist_radius_um': 15.0,
    },
    'fixed8': {
        'score': float(fixed['score']),
        'delta': float(fixed['delta_vs_control']),
        'edge_tp_fp_fn': [fixed.get('edge_tp'), fixed.get('edge_fp'), fixed.get('edge_fn')],
    },
    'holdout8': {
        'score': float(hold['score']),
        'delta': float(hold['delta_vs_control']),
        'edge_tp_fp_fn': [hold.get('edge_tp'), hold.get('edge_fp'), hold.get('edge_fn')],
    },
}
promote = (
    comparison['fixed8']['delta'] > 0
    and comparison['holdout8']['delta'] > 0
)
comparison['promote'] = bool(promote)
comparison['decision'] = 'PROMOTE' if promote else 'REJECT'
(OUT / 'comparison.json').write_text(json.dumps(comparison, indent=2, sort_keys=True) + '\n')
(OUT / 'DONE').write_text('ok\n')
if OUT_NFS.exists():
    shutil.rmtree(OUT_NFS)
shutil.copytree(OUT, OUT_NFS, ignore=shutil.ignore_patterns('work', 'support_repo'))
(OUT_NFS / 'DONE').write_text('ok\n')
print('COMPARISON', json.dumps(comparison, indent=2, sort_keys=True))
print('DECISION', comparison['decision'])
PY
" 2>&1 | tee -a "${LOG}"

MIRROR="${SRC}/outputs/experiments/margin_gated_dist_rank_v1"
mkdir -p "${MIRROR}"
srun -p gpu_batch -N1 -n1 --gres=gpu:0 -t 00:15:00 --mem=8G --chdir=/tmp bash -lc '
  ROOT=${HOME}/biohub-outputs/experiments/margin_gated_dist_rank_v1
  cd "$ROOT"
  tar cf - DONE comparison.json experiment_config.yaml \
    fixed8/DONE fixed8/summary.json fixed8/per_dataset.csv fixed8/command.txt \
    fixed8/experiment_config.yaml fixed8/predictions/postprocessed_submission.csv \
    holdout8/DONE holdout8/summary.json holdout8/per_dataset.csv holdout8/command.txt \
    holdout8/experiment_config.yaml holdout8/predictions/postprocessed_submission.csv
' | tar xf - -C "${MIRROR}"
echo "LOGIN_MIRROR_DONE ${MIRROR}"
find "${MIRROR}" -type f | sort | head -40
echo "Finished margin_gated_dist_rank_v1 at $(date -Iseconds)" | tee -a "${LOG}"
