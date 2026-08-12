#!/bin/bash
#SBATCH -p gpu_batch
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 0-04:00:00
#SBATCH -J edge045moff
#SBATCH -o /home/mc46451/biohub-cell-tracking/logs/slurm/edge_thresh_0_45_motion_off_%j.out
#SBATCH -e /home/mc46451/biohub-cell-tracking/logs/slurm/edge_thresh_0_45_motion_off_%j.err

# GPU: edge_threshold=0.45 under promoted motion-relink-OFF recipe.
# Control (edge=0.5, motion OFF): fixed-8 0.906725 / holdout 0.960307
set -euo pipefail

SRC=/home/mc46451/biohub-cell-tracking
FIXED8=(
  44b6_0113de3b 44b6_0b24845f 44b6_341df25f 44b6_e57ff5c6
  6bba_05b6850b 6bba_05db0fb1 6bba_969618f6 6bba_fc83837d
)
HOLDOUT8=(
  44b6_0c582fdc 44b6_0db75fae 44b6_12dfb391 44b6_144b256d
  6bba_062c8d37 6bba_07477033 6bba_07e24132 6bba_085bf656
)

DEST=/tmp/${USER}/biohub-edge-thresh-045
OUT_NFS=${HOME}/biohub-outputs/experiments/edge_thresh_0_45_motion_off_v1

echo "Starting edge_thresh 0.45 motion-off at $(date -Iseconds) on $(hostname)"

rm -rf "$DEST"
mkdir -p "$DEST"
cd "$SRC"
STAGE_INPUTS=()
for d in "${FIXED8[@]}" "${HOLDOUT8[@]}"; do
  STAGE_INPUTS+=("data/competition/train/${d}.zarr" "data/competition/train/${d}.geff")
done
tar \
  --exclude='.venv' --exclude='.git' --exclude='**/__pycache__' \
  --exclude='outputs' --exclude='logs' \
  --exclude='data/support/repo/predictions' --exclude='data/tmp_seed314159' \
  -cf - \
  README.md LICENSE pyproject.toml configs src scripts \
  data/support \
  "${STAGE_INPUTS[@]}" \
| tar xf - -C "$DEST"
cd "$DEST"

source "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate biohub
python -m pip install -U pip >/dev/null
python -m pip install --no-deps data/support/wheels/*.whl >/dev/null 2>&1 || true
python -m pip install --find-links=data/support/wheels --prefer-binary \
  tracksdata zarr "geff>=1.1.3.1.1" "geff-spec<1.2" "ilpy>=0.5.1" \
  pyscipopt polars blosc2 dask imagecodecs pyarrow "rustworkx>=0.17.1" \
  "sqlalchemy>=2" "scikit-image>=0.24" "numcodecs>=0.13,<0.16" donfig >/dev/null
export PYTHONPATH="${DEST}/src${PYTHONPATH:+:$PYTHONPATH}"

rm -rf "$OUT_NFS"
mkdir -p "$OUT_NFS"
export OUT_NFS

python - <<'PY'
from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path

from biohub_pipeline.config import load_config
from biohub_pipeline.evaluation import official_spec_summarise
from biohub_pipeline.fixed8_cv import (
    FIXED8_DATASETS,
    _copy_raw_predictions,
    evaluate_postprocessed_predictions,
    evaluate_tables,
    find_fixed8_prediction_geffs,
    write_metric_outputs,
    _estimated_total_nodes,
    _read_graph_tables,
)
from biohub_pipeline.inference import (
    apply_spatial_d4_patch,
    build_predict_command,
    run_prediction,
)
from biohub_pipeline.submission import write_submission_from_geff

DATA_DIR = Path('data/competition/train').resolve()
SUPPORT = Path('data/support').resolve()
CFG = Path('configs/experiments/edge_thresh_0_45_motion_off_det0_96875.yaml').resolve()
OUT = Path('outputs/experiments/edge_thresh_0_45_motion_off_v1').resolve()
OUT_NFS = Path(os.environ['OUT_NFS'])

CONTROL_FIXED = 0.9067252533426169
CONTROL_HOLD = 0.9603073913068769
HOLDOUT8 = [
    '44b6_0c582fdc', '44b6_0db75fae', '44b6_12dfb391', '44b6_144b256d',
    '6bba_062c8d37', '6bba_07477033', '6bba_07e24132', '6bba_085bf656',
]

config = load_config(CFG)
assert abs(float(config.inference['edge_threshold']) - 0.45) < 1e-12
assert config.postprocessing['output_motion_relink'] is False
assert abs(float(config.inference['detection_threshold']) - 0.96875) < 1e-12


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
    (out / 'command.txt').write_text(' '.join(command) + '\n')
    print('RUN', name, 'edge_threshold_flag', '--edge-threshold' in command, flush=True)
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
        'label': f'{name}_edge_thresh_0_45_motion_off',
        'edge_threshold': 0.45,
        'output_motion_relink': False,
        'detection_threshold': 0.96875,
        'delta_vs_motion_off_control': float(aggregate['score']) - control_score,
        'control_score_motion_off_edge_0_5': control_score,
        'runtime_seconds': runtime,
    }
    write_metric_outputs(
        out, rows, summary,
        {
            'schema_version': 1,
            'experiment': 'edge_thresh_0_45_motion_off_v1',
            'split': name,
            'config': str(CFG),
            'raw_geff': str(raw_dir),
            'note': 'GPU inference; motion_relink OFF; edge_threshold 0.45',
        },
    )
    (out / 'DONE').write_text('ok\n')
    print(name, 'score', summary['score'], 'delta', summary['delta_vs_motion_off_control'], flush=True)
    return summary


if OUT.exists():
    shutil.rmtree(OUT)
OUT.mkdir(parents=True)
shutil.copy2(CFG, OUT / 'experiment_config.yaml')

fixed = run_split('fixed8', list(FIXED8_DATASETS), CONTROL_FIXED)
hold = run_split('holdout8', HOLDOUT8, CONTROL_HOLD)
comparison = {
    'hypothesis': 'Lowering edge_threshold 0.5→0.45 under motion-relink OFF recovers near-threshold true edges net-positive',
    'control_fixed8_motion_off_edge_0_5': CONTROL_FIXED,
    'control_holdout8_motion_off_edge_0_5': CONTROL_HOLD,
    'fixed8': {
        'score': float(fixed['score']),
        'delta': float(fixed['delta_vs_motion_off_control']),
        'edge_tp_fp_fn': [fixed['edge_tp'], fixed['edge_fp'], fixed['edge_fn']],
    },
    'holdout8': {
        'score': float(hold['score']),
        'delta': float(hold['delta_vs_motion_off_control']),
        'edge_tp_fp_fn': [hold['edge_tp'], hold['edge_fp'], hold['edge_fn']],
    },
}
promote = (
    comparison['fixed8']['delta'] >= 0.001
    and comparison['holdout8']['delta'] >= -0.002
) or (
    comparison['fixed8']['delta'] > 0
    and comparison['holdout8']['delta'] > 0
    and comparison['fixed8']['delta'] + comparison['holdout8']['delta'] >= 0.001
)
comparison['promote'] = bool(promote)
comparison['decision'] = (
    'PROMOTE edge_threshold=0.45'
    if promote
    else 'KEEP edge_threshold=0.5; 0.45 not justified'
)
(OUT / 'comparison.json').write_text(json.dumps(comparison, indent=2, sort_keys=True) + '\n')
(OUT / 'DONE').write_text('ok\n')
if OUT_NFS.exists():
    shutil.rmtree(OUT_NFS)
shutil.copytree(OUT, OUT_NFS, ignore=shutil.ignore_patterns('work'))
print('COMPARISON', json.dumps(comparison, indent=2, sort_keys=True))
print('DECISION', comparison['decision'])
PY

echo "Job finished at $(date -Iseconds)"
