#!/usr/bin/env bash
# Probability-band diagnostic for the candidate-edge gate under motion-relink OFF.
# Uses saved candidate_capture + GT GEFFs + existing raw GEFFs / finals. No re-inference.
set -euo pipefail

SRC="${1:-/home/mc46451/biohub-cell-tracking}"
LOG_DIR="${SRC}/logs/slurm"
mkdir -p "${LOG_DIR}"
LOG="${LOG_DIR}/edge_gate_band_diagnostic.log"

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

DIAG_STAGE="${SRC}/.tmp_edge_gate_diags"
rm -rf "${DIAG_STAGE}"
mkdir -p "${DIAG_STAGE}"
cp -f \
  "${SRC}/outputs/experiments/candidate_edge_bottleneck_v1/candidate_edge_diagnostic.csv" \
  "${DIAG_STAGE}/fixed8_diagnostic.csv"
cp -f \
  "${SRC}/outputs/experiments/candidate_edge_bottleneck_holdout8/candidate_edge_diagnostic.csv" \
  "${DIAG_STAGE}/holdout8_diagnostic.csv"
cp -f \
  "${SRC}/outputs/experiments/bottleneck_rediagnose_motion_off_v1/fixed8_motion_off/candidate_edge_diagnostic.csv" \
  "${DIAG_STAGE}/fixed8_motion_off_diagnostic.csv" 2>/dev/null || true
cp -f \
  "${SRC}/outputs/experiments/bottleneck_rediagnose_motion_off_v1/holdout8_motion_off/candidate_edge_diagnostic.csv" \
  "${DIAG_STAGE}/holdout8_motion_off_diagnostic.csv" 2>/dev/null || true

echo "Starting edge-gate band diagnostic at $(date -Iseconds)" | tee "${LOG}"

tar \
  --exclude='.venv' --exclude='.git' --exclude='**/__pycache__' \
  --exclude='outputs' --exclude='logs' \
  --exclude='data/support/repo/predictions' --exclude='data/tmp_seed314159' \
  -cf - \
  README.md LICENSE pyproject.toml configs src scripts \
  data/support \
  .tmp_edge_gate_diags \
  "${STAGE_INPUTS[@]}" \
| srun -p gpu_batch -N1 -n1 --gres=gpu:1 --cpus-per-task=8 --mem=32G -t 0-01:00:00 \
  --chdir=/tmp \
  bash -lc '
set -euo pipefail
DEST=/tmp/${USER}/biohub-edge-gate-band
OUT_NFS=${HOME}/biohub-outputs/experiments/edge_gate_sweep_motion_off_v1
export OUT_NFS
export FIXED_CAP=${HOME}/biohub-outputs/experiments/candidate_edge_bottleneck_v1/candidate_capture
export HOLD_CAP=${HOME}/biohub-outputs/experiments/candidate_edge_bottleneck_holdout8/candidate_capture
export RAW50_F=${HOME}/biohub-outputs/experiments/candidate_edge_bottleneck_v1/raw_geff
export RAW50_H=${HOME}/biohub-outputs/holdout8/det0_96875_safeon/raw_geff
export RAW45_F=${HOME}/biohub-outputs/experiments/edge_thresh_0_45_motion_off_v1/fixed8/raw_geff
export RAW45_H=${HOME}/biohub-outputs/experiments/edge_thresh_0_45_motion_off_v1/holdout8/raw_geff
export RAW40_F=${HOME}/biohub-outputs/experiments/edge_thresh_0_40_motion_off_v1/fixed8/raw_geff
export RAW40_H=${HOME}/biohub-outputs/experiments/edge_thresh_0_40_motion_off_v1/holdout8/raw_geff
export FINAL50_F=${HOME}/biohub-outputs/experiments/ppstage_ablation_det0_96875/fixed8/ppstage_motion_relink_off_det0_96875/predictions/postprocessed_submission.csv
export FINAL50_H=${HOME}/biohub-outputs/experiments/ppstage_ablation_det0_96875/holdout8/ppstage_motion_relink_off_det0_96875/predictions/postprocessed_submission.csv
export FINAL45_F=${HOME}/biohub-outputs/experiments/edge_thresh_0_45_motion_off_v1/fixed8/predictions/postprocessed_submission.csv
export FINAL45_H=${HOME}/biohub-outputs/experiments/edge_thresh_0_45_motion_off_v1/holdout8/predictions/postprocessed_submission.csv
export FINAL40_F=${HOME}/biohub-outputs/experiments/edge_thresh_0_40_motion_off_v1/fixed8/predictions/postprocessed_submission.csv
export FINAL40_H=${HOME}/biohub-outputs/experiments/edge_thresh_0_40_motion_off_v1/holdout8/predictions/postprocessed_submission.csv

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

python scripts/analyze_edge_gate_bands.py
' 2>&1 | tee -a "${LOG}"

mkdir -p "${SRC}/outputs/experiments/edge_gate_sweep_motion_off_v1/band_diagnostic"
srun -p gpu_batch -N1 -n1 --gres=gpu:1 -t 00:08:00 --mem=4G --chdir=/tmp bash -lc '
  cd ${HOME}/biohub-outputs/experiments/edge_gate_sweep_motion_off_v1/band_diagnostic && \
  tar cf - DONE combined_band_summary.json fixed8_band_summary.json holdout8_band_summary.json
' | tar xf - -C "${SRC}/outputs/experiments/edge_gate_sweep_motion_off_v1/band_diagnostic"
rm -rf "${DIAG_STAGE}"
echo "BAND_DIAG_OK $(date -Iseconds)" | tee -a "${LOG}"
