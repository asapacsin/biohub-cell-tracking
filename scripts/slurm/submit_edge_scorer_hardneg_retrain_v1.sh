#!/usr/bin/env bash
# Login-node submitter. Stages code+data to compute NFS, then sbatch the GPU worker.
# Safe to nohup: staging may take 15-30 min; sbatch is then disconnect-safe.
set -euo pipefail

SRC="${1:-/home/mc46451/biohub-cell-tracking}"
LOG_DIR="${SRC}/logs/slurm"
mkdir -p "${LOG_DIR}"
LOG="${LOG_DIR}/submit_edge_scorer_hardneg_retrain_v1.log"

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

GIT=$(git -C "${SRC}" rev-parse HEAD 2>/dev/null || echo unknown)
export HNRETR_GIT="${GIT}"

{
  echo "Starting submit edge_scorer_hardneg_retrain_v1 at $(date -Iseconds)"
  echo "ETA_WINDOW_OPEN=$(date -Iseconds) STAGE_MIN=25 ETA_HOURS=14"
  echo "GIT=${GIT}"
  squeue -u "${USER}" || true
} | tee "${LOG}"

if squeue -u "${USER}" -h -n hnretrv1 | grep -q .; then
  echo "ERROR: hnretrv1 already queued/running; not duplicating" | tee -a "${LOG}"
  squeue -u "${USER}" -n hnretrv1 | tee -a "${LOG}"
  exit 3
fi

echo "Preflight compute NFS capture + empty output dir" | tee -a "${LOG}"
srun -p gpu_batch -N1 -n1 --gres=gpu:0 --cpus-per-task=2 --mem=4G -t 00:05:00 --chdir=/tmp \
  bash -lc '
set -euo pipefail
CAP=${HOME}/biohub-outputs/experiments/candidate_capture_recipe_c_edge_0_40_v1
OUT=${HOME}/biohub-outputs/experiments/edge_scorer_hardneg_retrain_v1
mkdir -p ${HOME}/biohub-outputs/stage ${HOME}/biohub-outputs/logs
test -d ${CAP}/fixed8/candidate_capture
test -f ${CAP}/fixed8/candidate_edge_diagnostic.csv
test -f ${CAP}/fixed8/metric_by_dataset.csv
test -f ${CAP}/fixed8/score_summary.json
test -f ${CAP}/holdout8/candidate_edge_diagnostic.csv
test -f ${CAP}/holdout8/metric_by_dataset.csv
test -f ${CAP}/holdout8/score_summary.json
if [[ -e ${OUT} ]]; then
  echo "ERROR: output already exists: ${OUT}" >&2
  exit 2
fi
echo PREFLIGHT_OK
' 2>&1 | tee -a "${LOG}"

echo "Staging tarball to compute NFS (~7G code+support+16 zarr/geff)" | tee -a "${LOG}"
tar \
  --exclude='.venv' --exclude='.git' --exclude='**/__pycache__' \
  --exclude='outputs' --exclude='logs' \
  --exclude='data/support/repo/predictions' --exclude='data/tmp_seed314159' \
  -cf - \
  README.md LICENSE pyproject.toml configs src scripts \
  data/support \
  "${STAGE_INPUTS[@]}" \
| srun -p gpu_batch -N1 -n1 --gres=gpu:0 --cpus-per-task=4 --mem=16G -t 0-01:00:00 --chdir=/tmp \
  bash -lc 'cat > ${HOME}/biohub-outputs/stage/edge_scorer_hardneg_retrain_v1.tar && ls -lh ${HOME}/biohub-outputs/stage/edge_scorer_hardneg_retrain_v1.tar' \
  2>&1 | tee -a "${LOG}"

echo "Submitting GPU worker" | tee -a "${LOG}"
JOB=$(sbatch --export=ALL,HNRETR_GIT="${GIT}" \
  "${SRC}/scripts/slurm/run_edge_scorer_hardneg_retrain_v1.sbatch" | awk '{print $4}')
echo "SUBMITTED slurm:${JOB}" | tee -a "${LOG}"
echo "${JOB}" > "${LOG_DIR}/edge_scorer_hardneg_retrain_v1.jobid"
squeue -j "${JOB}" | tee -a "${LOG}"
echo "Finished submit at $(date -Iseconds); GPU job is disconnect-safe" | tee -a "${LOG}"
