#!/usr/bin/env bash
# Re-run candidate-edge cause taxonomy against motion-relink-OFF finals.
# No re-inference: reuse candidate_capture + raw GEFFs + saved postprocessed CSVs.
set -euo pipefail

SRC="${1:-/home/mc46451/biohub-cell-tracking}"
LOG_DIR="${SRC}/logs/slurm"
mkdir -p "${LOG_DIR}"
LOG="${LOG_DIR}/rediagnose_motion_off.log"

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
  STAGE_INPUTS+=("data/competition/train/${d}.geff")
done

echo "Starting motion-off rediagnosis at $(date -Iseconds)" | tee "${LOG}"

tar \
  --exclude='.venv' --exclude='.git' --exclude='**/__pycache__' \
  --exclude='outputs' --exclude='logs' \
  --exclude='data/support/repo/predictions' --exclude='data/tmp_seed314159' \
  -cf - \
  README.md LICENSE pyproject.toml configs src scripts \
  data/support \
  "${STAGE_INPUTS[@]}" \
| srun -p gpu_batch -N1 -n1 --gres=gpu:1 --cpus-per-task=4 --mem=24G -t 0-00:40:00 \
  --chdir=/tmp \
  bash -lc "
set -euo pipefail
DEST=/tmp/\${USER}/biohub-rediag-motion-off
OUT_NFS=\${HOME}/biohub-outputs/experiments/bottleneck_rediagnose_motion_off_v1
FIXED_RAW=\${HOME}/biohub-outputs/experiments/candidate_edge_bottleneck_v1/raw_geff
HOLD_RAW=\${HOME}/biohub-outputs/holdout8/det0_96875_safeon/raw_geff
if [[ ! -d \"\$HOLD_RAW\" ]]; then
  HOLD_RAW=\${HOME}/biohub-outputs/experiments/candidate_edge_bottleneck_holdout8/raw_geff
fi
FIXED_CAP=\${HOME}/biohub-outputs/experiments/candidate_edge_bottleneck_v1/candidate_capture
HOLD_CAP=\${HOME}/biohub-outputs/experiments/candidate_edge_bottleneck_holdout8/candidate_capture
FIXED_PRED=\${HOME}/biohub-outputs/experiments/ppstage_ablation_det0_96875/fixed8/ppstage_motion_relink_off_det0_96875/predictions/postprocessed_submission.csv
HOLD_PRED=\${HOME}/biohub-outputs/experiments/ppstage_ablation_det0_96875/holdout8/ppstage_motion_relink_off_det0_96875/predictions/postprocessed_submission.csv
FIXED_CTRL_PRED=\${HOME}/biohub-outputs/experiments/ppstage_ablation_det0_96875/fixed8/ppstage_control_det0_96875/predictions/postprocessed_submission.csv
HOLD_CTRL_PRED=\${HOME}/biohub-outputs/experiments/ppstage_ablation_det0_96875/holdout8/ppstage_control_det0_96875/predictions/postprocessed_submission.csv

rm -rf \"\$DEST\"
mkdir -p \"\$DEST\"
tar xf - -C \"\$DEST\"
cd \"\$DEST\"

source \"\${HOME}/miniconda3/etc/profile.d/conda.sh\"
conda activate biohub
python -m pip install -U pip >/dev/null
python -m pip install --no-deps data/support/wheels/*.whl >/dev/null 2>&1 || true
python -m pip install --find-links=data/support/wheels --prefer-binary \
  tracksdata zarr \"geff>=1.1.3.1.1\" \"geff-spec<1.2\" pandas numpy pyarrow >/dev/null
export PYTHONPATH=\"\${DEST}/src\${PYTHONPATH:+:\$PYTHONPATH}\"

for p in \"\$FIXED_RAW\" \"\$HOLD_RAW\" \"\$FIXED_CAP\" \"\$HOLD_CAP\" \"\$FIXED_PRED\" \"\$HOLD_PRED\"; do
  test -e \"\$p\" || { echo MISSING \"\$p\"; exit 1; }
done
rm -rf \"\$OUT_NFS\"
mkdir -p \"\$OUT_NFS\"

export FIXED_RAW HOLD_RAW FIXED_CAP HOLD_CAP FIXED_PRED HOLD_PRED FIXED_CTRL_PRED HOLD_CTRL_PRED OUT_NFS
python - <<'PY'
from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

from biohub_pipeline.association_density import read_geff_tables
from biohub_pipeline.candidate_bottleneck import analyze_dataset, classify, summarize
from biohub_pipeline.config import load_config

FIXED8 = [
    '44b6_0113de3b', '44b6_0b24845f', '44b6_341df25f', '44b6_e57ff5c6',
    '6bba_05b6850b', '6bba_05db0fb1', '6bba_969618f6', '6bba_fc83837d',
]
HOLDOUT8 = [
    '44b6_0c582fdc', '44b6_0db75fae', '44b6_12dfb391', '44b6_144b256d',
    '6bba_062c8d37', '6bba_07477033', '6bba_07e24132', '6bba_085bf656',
]

DATA_DIR = Path('data/competition/train')
CFG = Path('configs/experiments/recipe_c_motion_relink_off_det0_96875.yaml')
config = load_config(CFG)
scale = tuple(float(v) for v in config.postprocessing['voxel_scale_um'])
max_um = float(config.local_cv['max_match_um'])
OUT = Path(os.environ['OUT_NFS'])


def run_split(name, datasets, raw_dir, cap_dir, pred_csv):
    preds = pd.read_csv(pred_csv)
    frames, metadata = [], []
    for dataset in datasets:
        frame, meta = analyze_dataset(
            dataset,
            Path(cap_dir) / dataset,
            Path(raw_dir) / f'{dataset}.geff',
            read_geff_tables(DATA_DIR / f'{dataset}.geff'),
            preds,
            scale=scale,
            max_match_um=max_um,
        )
        frames.append(frame)
        metadata.append(meta)
        print(name, dataset, 'errors', int((frame.cause != 'correct').sum()), flush=True)
    diagnostic = pd.concat(frames, ignore_index=True)
    summary, by_dataset = summarize(diagnostic, metadata)
    decision = classify(summary)
    out = OUT / name
    out.mkdir(parents=True, exist_ok=True)
    diagnostic.to_csv(out / 'candidate_edge_diagnostic.csv', index=False)
    summary.to_csv(out / 'cause_summary.csv', index=False)
    by_dataset.to_csv(out / 'cause_by_dataset.csv', index=False)
    (out / 'decision.json').write_text(json.dumps(decision, indent=2, sort_keys=True) + '\n')
    overall = summary[(summary.group_type == 'overall') & (summary.group == 'all')].iloc[0]
    compact = {
        'split': name,
        'classification': decision['classification'],
        'mechanism_shares': decision['mechanism_shares'],
        'causal_errors': int(decision['causal_errors']),
        'ranking_share': float(decision['ranking_share']),
        'counts': {
            c: int(overall[c])
            for c in (
                'correct', 'detection_miss', 'scorer_ranking', 'candidate_threshold',
                'ilp_global', 'postprocessing_removed', 'postprocessing_rematch',
            )
        },
        'recommended_next_action': decision['recommended_next_action'],
    }
    (out / 'compact.json').write_text(json.dumps(compact, indent=2, sort_keys=True) + '\n')
    return compact


fixed = run_split(
    'fixed8_motion_off',
    FIXED8,
    os.environ['FIXED_RAW'],
    os.environ['FIXED_CAP'],
    os.environ['FIXED_PRED'],
)
hold = run_split(
    'holdout8_motion_off',
    HOLDOUT8,
    os.environ['HOLD_RAW'],
    os.environ['HOLD_CAP'],
    os.environ['HOLD_PRED'],
)
# Also control for delta comparison of cause counts
fixed_ctrl = run_split(
    'fixed8_control',
    FIXED8,
    os.environ['FIXED_RAW'],
    os.environ['FIXED_CAP'],
    os.environ['FIXED_CTRL_PRED'],
)
hold_ctrl = run_split(
    'holdout8_control',
    HOLDOUT8,
    os.environ['HOLD_RAW'],
    os.environ['HOLD_CAP'],
    os.environ['HOLD_CTRL_PRED'],
)

combined = {
    'fixed8_motion_off': fixed,
    'holdout8_motion_off': hold,
    'fixed8_control': fixed_ctrl,
    'holdout8_control': hold_ctrl,
    'delta_counts_fixed8': {
        k: fixed['counts'][k] - fixed_ctrl['counts'][k] for k in fixed['counts']
    },
    'delta_counts_holdout8': {
        k: hold['counts'][k] - hold_ctrl['counts'][k] for k in hold['counts']
    },
}
(OUT / 'combined_summary.json').write_text(json.dumps(combined, indent=2, sort_keys=True) + '\n')
(OUT / 'DONE').write_text('ok\n')
print(json.dumps(combined, indent=2, sort_keys=True))
PY
" 2>&1 | tee -a "${LOG}"

mkdir -p "${SRC}/outputs/experiments/bottleneck_rediagnose_motion_off_v1"
srun -p gpu_batch -N1 -n1 --gres=gpu:1 -t 00:08:00 --mem=4G --chdir=/tmp bash -lc '
  cd ${HOME}/biohub-outputs/experiments/bottleneck_rediagnose_motion_off_v1 && \
  tar cf - DONE combined_summary.json \
    fixed8_motion_off/compact.json fixed8_motion_off/cause_summary.csv fixed8_motion_off/decision.json \
    holdout8_motion_off/compact.json holdout8_motion_off/cause_summary.csv holdout8_motion_off/decision.json \
    fixed8_control/compact.json holdout8_control/compact.json \
    fixed8_motion_off/candidate_edge_diagnostic.csv holdout8_motion_off/candidate_edge_diagnostic.csv
' | tar xf - -C "${SRC}/outputs/experiments/bottleneck_rediagnose_motion_off_v1"
echo "LOGIN_MIRROR_OK $(date -Iseconds)" | tee -a "${LOG}"
cat "${SRC}/outputs/experiments/bottleneck_rediagnose_motion_off_v1/combined_summary.json"
