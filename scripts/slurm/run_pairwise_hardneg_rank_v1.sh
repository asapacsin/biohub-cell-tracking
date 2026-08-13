#!/usr/bin/env bash
# GPU eval: appearance-aware pairwise hard-neg ranking under frozen recipe_c edge=0.40 motion OFF.
# Does not change motion_relink / edge_threshold / det / ensemble / short-track.
# Control: fixed-8 0.9181439782806684 / holdout-8 0.9646726188580379
# ETA: ~2.5–3.5h wall on 1×GPU (same shape as margin_gated_dist_rank_v1)
set -euo pipefail

SRC="${1:-/home/mc46451/biohub-cell-tracking}"
LOG_DIR="${SRC}/logs/slurm"
mkdir -p "${LOG_DIR}"
LOG="${LOG_DIR}/pairwise_hardneg_rank_v1.log"

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

echo "Starting pairwise_hardneg_rank_v1 at $(date -Iseconds)" | tee "${LOG}"
echo "ETA_WINDOW_OPEN=$(date -Iseconds) ETA_HOURS=3.5" | tee -a "${LOG}"

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
DEST=/tmp/\${USER}/biohub-pairwise-hardneg-v1
OUT_NFS=\${HOME}/biohub-outputs/experiments/pairwise_hardneg_rank_v1
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
    apply_spatial_d4_patch,
    build_predict_command,
    run_prediction,
)
from biohub_pipeline.submission import write_submission_from_geff

DATA_DIR = Path('data/competition/train').resolve()
SUPPORT = Path('data/support').resolve()
CFG = Path('configs/experiments/recipe_c_edge_0_40_pairwise_hardneg_v1.yaml').resolve()
OUT = Path('outputs/experiments/pairwise_hardneg_rank_v1').resolve()
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
assert config.inference.get('margin_gated_dist_lambda') in (None, 0, 0.0)
assert len(config.inference['pairwise_hardneg_w']) == 6
print(
    'RECIPE_OK edge=', config.inference['edge_threshold'],
    'motion=', config.postprocessing['output_motion_relink'],
    'pairwise_dens_min=', config.inference['pairwise_hardneg_dens_min'],
    'pairwise_gap_max=', config.inference['pairwise_hardneg_gap_max'],
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
    (out / 'command.txt').write_text(' '.join(command) + '\n')
    print(
        'RUN', name,
        'has_pairwise', '--pairwise-hardneg-weights' in command,
        'has_edge_threshold', '--edge-threshold' in command,
        'has_margin', '--margin-gated-dist-lambda' in command,
        flush=True,
    )
    assert '--pairwise-hardneg-weights' in command
    assert '--edge-threshold' in command
    assert '--margin-gated-dist-lambda' not in command
    started = time.perf_counter()
    run_prediction(command, repo)
    geffs = find_fixed8_prediction_geffs(repo, str(config.inference['method']), datasets)
    _copy_raw_predictions(geffs, out, config_path=CFG)
    pred_csv = out / 'predictions' / 'postprocessed_submission.csv'
    write_submission_from_geff(geffs, config, DATA_DIR, pred_csv)
    if datasets == list(FIXED8_DATASETS):
        rows, aggregate = evaluate_postprocessed_predictions(pred_csv, DATA_DIR, config)
    else:
        rows, aggregate = evaluate_custom(pred_csv, datasets)
    runtime = time.perf_counter() - started
    summary = {
        **aggregate,
        'label': f'{name}_pairwise_hardneg_rank_v1',
        'edge_threshold': 0.40,
        'output_motion_relink': False,
        'detection_threshold': 0.96875,
        'pairwise_hardneg_dens_min': float(config.inference['pairwise_hardneg_dens_min']),
        'pairwise_hardneg_gap_max': float(config.inference['pairwise_hardneg_gap_max']),
        'pairwise_hardneg_w': [float(x) for x in config.inference['pairwise_hardneg_w']],
        'delta_vs_control': float(aggregate['score']) - control_score,
        'control_score': control_score,
        'runtime_seconds': runtime,
    }
    write_metric_outputs(
        out, rows, summary,
        {
            'schema_version': 1,
            'experiment': 'pairwise_hardneg_rank_v1',
            'split': name,
            'config': str(CFG),
            'frozen_recipe': 'recipe_c_motion_off_edge_0_40_det0_96875',
        },
    )
    # compact mirror to NFS (drop work/support_repo)
    nfs_split = OUT_NFS / name
    if nfs_split.exists():
        shutil.rmtree(nfs_split)
    nfs_split.mkdir(parents=True)
    for item in out.iterdir():
        if item.name == 'work':
            continue
        dest = nfs_split / item.name
        if item.is_dir():
            shutil.copytree(item, dest)
        else:
            shutil.copy2(item, dest)
    (out / 'DONE').write_text('ok\n')
    (nfs_split / 'DONE').write_text('ok\n')
    print('DONE', name, 'score', summary['score'], 'delta', summary['delta_vs_control'], flush=True)
    return summary

fixed = run_split('fixed8', list(FIXED8_DATASETS), CONTROL_FIXED)
hold = run_split('holdout8', HOLDOUT8, CONTROL_HOLD)
comparison = {
    'experiment': 'pairwise_hardneg_rank_v1',
    'control_fixed': CONTROL_FIXED,
    'control_holdout': CONTROL_HOLD,
    'fixed8_score': fixed['score'],
    'holdout8_score': hold['score'],
    'fixed8_delta': fixed['delta_vs_control'],
    'holdout8_delta': hold['delta_vs_control'],
    'decision_hint': (
        'PROMOTE' if fixed['delta_vs_control'] > 0 and hold['delta_vs_control'] > 0
        else 'REJECT' if fixed['delta_vs_control'] < 0 or hold['delta_vs_control'] < 0
        else 'INCONCLUSIVE'
    ),
}
(OUT / 'comparison.json').write_text(json.dumps(comparison, indent=2), encoding='utf-8')
(OUT_NFS / 'comparison.json').write_text(json.dumps(comparison, indent=2), encoding='utf-8')
(OUT / 'DONE').write_text('ok\n')
(OUT_NFS / 'DONE').write_text('ok\n')
print('COMPARISON', json.dumps(comparison), flush=True)
PY

echo \"GPU pairwise_hardneg_rank_v1 finished at \$(date -Iseconds)\"
" 2>&1 | tee -a "${LOG}"

echo "Mirroring compact artifacts to login outputs/ ..." | tee -a "${LOG}"
mkdir -p "${SRC}/outputs/experiments/pairwise_hardneg_rank_v1"
srun -p gpu_batch -N1 -n1 --gres=gpu:0 --cpus-per-task=2 --mem=8G -t 0-00:20:00 --chdir=/tmp \
  bash -lc 'cd ${HOME}/biohub-outputs/experiments/pairwise_hardneg_rank_v1 && tar -czf - comparison.json DONE fixed8/score_summary.json fixed8/metric_by_dataset.csv fixed8/command.txt fixed8/DONE holdout8/score_summary.json holdout8/metric_by_dataset.csv holdout8/command.txt holdout8/DONE 2>/dev/null || tar -czf - comparison.json DONE fixed8 holdout8 --exclude="*/predictions/*" --exclude="*/raw_geff/*"' \
  > /tmp/pairwise_hardneg_mirror.tar.gz 2>>"${LOG}" || true
if [[ -s /tmp/pairwise_hardneg_mirror.tar.gz ]]; then
  tar -xzf /tmp/pairwise_hardneg_mirror.tar.gz -C "${SRC}/outputs/experiments/pairwise_hardneg_rank_v1" || true
fi
echo "Finished pairwise_hardneg_rank_v1 at $(date -Iseconds)" | tee -a "${LOG}"
