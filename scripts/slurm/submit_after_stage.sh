#!/usr/bin/env bash
# Wait for local stage_to_nfs.log to report STAGE_OK, then sbatch train_infer.
set -euo pipefail

LOGIN_ROOT="/home/mc46451/biohub-cell-tracking"
STAGE_LOG="${LOGIN_ROOT}/logs/slurm/stage_to_nfs.log"
LOGDIR="${LOGIN_ROOT}/logs/slurm"
mkdir -p "${LOGDIR}"

echo "Waiting for STAGE_OK in ${STAGE_LOG}"
for i in $(seq 1 360); do
  if grep -q 'STAGE_OK' "${STAGE_LOG}" 2>/dev/null; then
    echo "Stage finished at $(date -Iseconds)"
    break
  fi
  if ! pgrep -f 'stage_to_nfs.sh' >/dev/null 2>&1 && ! grep -q 'STAGE_OK' "${STAGE_LOG}" 2>/dev/null; then
    # Staging script ended without success marker.
    if grep -qi 'error\|failed\|No space' "${STAGE_LOG}" 2>/dev/null; then
      echo "ERROR: staging failed; see ${STAGE_LOG}" >&2
      exit 1
    fi
  fi
  echo "  waiting (${i}) $(date -Iseconds)"
  sleep 60
done

if ! grep -q 'STAGE_OK' "${STAGE_LOG}"; then
  echo "ERROR: STAGE_OK never appeared" >&2
  exit 1
fi

# Submit with chdir on NFS via a wrapper: sbatch reads the script from a path
# the batch node can see. Copy script content through stdin isn't supported, so
# use sbatch --chdir after confirming NFS tree exists with a short srun.
NFS_OK=$(srun -p gpu_batch -N 1 -n 1 --gres=gpu:1 -t 00:03:00 --mem=2G \
  bash -lc 'test -f ${HOME}/biohub-cell-tracking/.nfs_stage_complete && echo yes || echo no')
echo "NFS_OK=${NFS_OK}"
[[ "${NFS_OK}" == "yes" ]] || { echo "NFS tree missing" >&2; exit 1; }

JOBID=$(srun -p gpu_batch -N 1 -n 1 --gres=gpu:1 -t 00:05:00 --mem=2G bash -lc '
  cd "${HOME}/biohub-cell-tracking"
  mkdir -p logs/slurm
  sbatch --parsable scripts/slurm/train_infer.sbatch
')
echo "Submitted train_infer jobid=${JOBID}"
echo "${JOBID}" | tee "${LOGDIR}/train_infer.jobid"
