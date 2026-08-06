#!/usr/bin/env bash
set -euo pipefail
LOGIN_ROOT="/home/mc46451/biohub-cell-tracking"
DEST="${LOGIN_ROOT}/outputs/kaggle_submission"
LOG="${LOGIN_ROOT}/logs/slurm/fetch_v106.log"
mkdir -p "${DEST}" "$(dirname "${LOG}")"
echo "Waiting for NFS biohub-outputs/v106/DONE ..." | tee "${LOG}"
for i in $(seq 1 360); do
  if srun -p gpu_batch -N1 -n1 --gres=gpu:1 -t 00:03:00 --mem=2G \
      bash -lc 'test -f ${HOME}/biohub-outputs/v106/DONE && cat ${HOME}/biohub-outputs/v106/DONE' \
      >"${LOGIN_ROOT}/logs/slurm/v106_done_marker.txt" 2>/dev/null; then
    echo "Found DONE $(cat "${LOGIN_ROOT}/logs/slurm/v106_done_marker.txt")" | tee -a "${LOG}"
    break
  fi
  if pgrep -f 'scripts/slurm/run_v106_infer.sh' >/dev/null 2>&1 || squeue -u "$USER" -h 2>/dev/null | grep -q .; then
    echo "  still running (${i}) $(date -Iseconds)" | tee -a "${LOG}"
    sleep 90
  else
    echo "  polling NFS (${i}) $(date -Iseconds)" | tee -a "${LOG}"
    sleep 60
  fi
done
if [[ ! -s "${LOGIN_ROOT}/logs/slurm/v106_done_marker.txt" ]]; then
  echo "ERROR: V106 DONE never appeared" | tee -a "${LOG}"
  exit 1
fi
srun -p gpu_batch -N1 -n1 --gres=gpu:1 -t 00:10:00 --mem=4G bash -lc '
  cd ${HOME}/biohub-outputs/v106 && tar cf - submission_v106.csv DONE
' | tar xf - -C "${LOGIN_ROOT}/outputs/"
cp -f "${LOGIN_ROOT}/outputs/submission_v106.csv" "${DEST}/submission_v106.csv"
# Also stage as candidate next to classical
ls -la "${DEST}/submission_v106.csv" | tee -a "${LOG}"
echo "FETCH_OK" | tee -a "${LOG}"
