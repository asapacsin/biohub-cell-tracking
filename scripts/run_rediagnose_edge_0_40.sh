#!/usr/bin/env bash
# Re-run candidate-edge cause taxonomy against edge_threshold=0.40 + motion-relink OFF finals.
# Reuses candidate_capture (scores unchanged by gate) + edge-0.40 raw GEFFs/finals. No re-inference.
set -euo pipefail

SRC="${1:-/home/mc46451/biohub-cell-tracking}"
LOG_DIR="${SRC}/logs/slurm"
mkdir -p "${LOG_DIR}"
LOG="${LOG_DIR}/rediagnose_edge_0_40.log"

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

echo "Starting edge-0.40 rediagnosis at $(date -Iseconds)" | tee "${LOG}"

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
  bash -lc '
set -euo pipefail
DEST=/tmp/${USER}/biohub-rediag-edge040
OUT_NFS=${HOME}/biohub-outputs/experiments/bottleneck_rediagnose_edge_0_40_v1
FIXED_RAW=${HOME}/biohub-outputs/experiments/edge_thresh_0_40_motion_off_v1/fixed8/raw_geff
HOLD_RAW=${HOME}/biohub-outputs/experiments/edge_thresh_0_40_motion_off_v1/holdout8/raw_geff
FIXED_CAP=${HOME}/biohub-outputs/experiments/candidate_edge_bottleneck_v1/candidate_capture
HOLD_CAP=${HOME}/biohub-outputs/experiments/candidate_edge_bottleneck_holdout8/candidate_capture
FIXED_PRED=${HOME}/biohub-outputs/experiments/edge_thresh_0_40_motion_off_v1/fixed8/predictions/postprocessed_submission.csv
HOLD_PRED=${HOME}/biohub-outputs/experiments/edge_thresh_0_40_motion_off_v1/holdout8/predictions/postprocessed_submission.csv
# Compare against motion-off edge 0.5 finals
FIXED_CTRL_PRED=${HOME}/biohub-outputs/experiments/ppstage_ablation_det0_96875/fixed8/ppstage_motion_relink_off_det0_96875/predictions/postprocessed_submission.csv
HOLD_CTRL_PRED=${HOME}/biohub-outputs/experiments/ppstage_ablation_det0_96875/holdout8/ppstage_motion_relink_off_det0_96875/predictions/postprocessed_submission.csv
FIXED_CTRL_RAW=${HOME}/biohub-outputs/experiments/candidate_edge_bottleneck_v1/raw_geff
HOLD_CTRL_RAW=${HOME}/biohub-outputs/holdout8/det0_96875_safeon/raw_geff

rm -rf "$DEST"
mkdir -p "$DEST"
tar xf - -C "$DEST"
cd "$DEST"

source "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate biohub
python -m pip install -U pip >/dev/null
python -m pip install --no-deps data/support/wheels/*.whl >/dev/null 2>&1 || true
python -m pip install --find-links=data/support/wheels --prefer-binary \
  tracksdata zarr "geff>=1.1.3.1.1" "geff-spec<1.2" pandas numpy pyarrow >/dev/null
export PYTHONPATH="${DEST}/src${PYTHONPATH:+:$PYTHONPATH}"

for p in "$FIXED_RAW" "$HOLD_RAW" "$FIXED_CAP" "$HOLD_CAP" "$FIXED_PRED" "$HOLD_PRED"; do
  test -e "$p" || { echo MISSING "$p"; exit 1; }
done
rm -rf "$OUT_NFS"
mkdir -p "$OUT_NFS"

export FIXED_RAW HOLD_RAW FIXED_CAP HOLD_CAP FIXED_PRED HOLD_PRED
export FIXED_CTRL_PRED HOLD_CTRL_PRED FIXED_CTRL_RAW HOLD_CTRL_RAW OUT_NFS

python scripts/rediagnose_edge_threshold.py
' 2>&1 | tee -a "${LOG}"

mkdir -p "${SRC}/outputs/experiments/bottleneck_rediagnose_edge_0_40_v1"
srun -p gpu_batch -N1 -n1 --gres=gpu:1 -t 00:08:00 --mem=4G --chdir=/tmp bash -lc '
  cd ${HOME}/biohub-outputs/experiments/bottleneck_rediagnose_edge_0_40_v1 && \
  tar cf - DONE combined_summary.json \
    fixed8_edge_0_40/compact.json fixed8_edge_0_40/cause_summary.csv fixed8_edge_0_40/decision.json \
    holdout8_edge_0_40/compact.json holdout8_edge_0_40/cause_summary.csv holdout8_edge_0_40/decision.json \
    fixed8_edge_0_50_motion_off/compact.json holdout8_edge_0_50_motion_off/compact.json \
    fixed8_edge_0_40/candidate_edge_diagnostic.csv holdout8_edge_0_40/candidate_edge_diagnostic.csv
' | tar xf - -C "${SRC}/outputs/experiments/bottleneck_rediagnose_edge_0_40_v1"
echo "LOGIN_MIRROR_OK $(date -Iseconds)" | tee -a "${LOG}"
cat "${SRC}/outputs/experiments/bottleneck_rediagnose_edge_0_40_v1/combined_summary.json"
