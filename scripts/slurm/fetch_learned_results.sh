#!/usr/bin/env bash
# Poll NFS $HOME/biohub-outputs/DONE via srun and copy learned submission to login.
set -euo pipefail
LOGIN_ROOT="/home/mc46451/biohub-cell-tracking"
DEST="${LOGIN_ROOT}/outputs/kaggle_submission"
LOG="${LOGIN_ROOT}/logs/slurm/fetch_learned.log"
mkdir -p "${DEST}" "$(dirname "${LOG}")"

echo "Waiting for NFS biohub-outputs/DONE ..." | tee "${LOG}"
for i in $(seq 1 480); do
  if srun -p gpu_batch -N1 -n1 --gres=gpu:1 -t 00:03:00 --mem=2G \
      bash -lc 'test -f ${HOME}/biohub-outputs/DONE && cat ${HOME}/biohub-outputs/DONE' \
      >"${LOGIN_ROOT}/logs/slurm/done_marker.txt" 2>/dev/null; then
    echo "Found DONE $(cat "${LOGIN_ROOT}/logs/slurm/done_marker.txt")" | tee -a "${LOG}"
    break
  fi
  # Prefer not fighting the long stage_and_train job: only poll when it is gone.
  if squeue -u "$USER" -h -n bash 2>/dev/null | grep -q .; then
    echo "  train job still running; sleep (${i}) $(date -Iseconds)" | tee -a "${LOG}"
    sleep 120
  else
    echo "  polling NFS (${i}) $(date -Iseconds)" | tee -a "${LOG}"
    sleep 60
  fi
done

if [[ ! -s "${LOGIN_ROOT}/logs/slurm/done_marker.txt" ]]; then
  echo "ERROR: DONE never appeared" | tee -a "${LOG}"
  exit 1
fi

srun -p gpu_batch -N1 -n1 --gres=gpu:1 -t 00:10:00 --mem=4G bash -lc '
  set -e
  cd ${HOME}/biohub-outputs
  tar cf - submission artifacts DONE
' | tar xf - -C "${LOGIN_ROOT}/outputs/"
# Flatten into kaggle_submission
cp -f "${LOGIN_ROOT}/outputs/submission/submission_learned.csv" \
  "${DEST}/submission_learned.csv"
# Also keep classical as default submission.csv
echo "Fetched learned submission to ${DEST}/submission_learned.csv" | tee -a "${LOG}"
ls -la "${DEST}" | tee -a "${LOG}"
