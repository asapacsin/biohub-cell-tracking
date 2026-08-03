#!/usr/bin/env bash
# Durable Kaggle competition download + unzip. Safe to re-run; resumes partial zip.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

COMP="biohub-cell-tracking-during-development"
OUT_DIR="$ROOT/data/competition"
ZIP="$OUT_DIR/${COMP}.zip"
LOG="$ROOT/outputs/kaggle_full_download.log"
PID_FILE="$ROOT/outputs/kaggle_download.pid"
STATUS_FILE="$ROOT/outputs/kaggle_download.status"

mkdir -p "$OUT_DIR" "$ROOT/outputs"
export KAGGLE_CONFIG_DIR="$ROOT/.kaggle"
export KAGGLE_API_TOKEN
KAGGLE_API_TOKEN="$(python3 -c 'import json; print(json.load(open(".kaggle/kaggle.json"))["key"])')"

if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  echo "already_running pid=$(cat "$PID_FILE")" | tee "$STATUS_FILE"
  exit 0
fi

# Prefer project venv kaggle if present.
if [[ -x "$ROOT/.venv/bin/kaggle" ]]; then
  KAGGLE="$ROOT/.venv/bin/kaggle"
else
  KAGGLE="kaggle"
fi

{
  echo "started_at=$(date -Iseconds)"
  echo "target=$ZIP"
} > "$STATUS_FILE"

# Download (kaggle resumes incomplete zip via .kaggle-partial).
"$KAGGLE" competitions download -c "$COMP" -p "$OUT_DIR" >>"$LOG" 2>&1
echo "download_finished_at=$(date -Iseconds)" >>"$STATUS_FILE"

if [[ -f "$ZIP" ]]; then
  echo "unzip_started_at=$(date -Iseconds)" >>"$STATUS_FILE"
  python3 - <<PY
from pathlib import Path
import zipfile
zip_path = Path(${ZIP@Q})
with zipfile.ZipFile(zip_path) as zf:
    zf.extractall(zip_path.parent)
zip_path.unlink(missing_ok=True)
partial = zip_path.with_name(zip_path.name + ".kaggle-partial")
partial.unlink(missing_ok=True)
print("unzip complete")
PY
  echo "unzip_finished_at=$(date -Iseconds)" >>"$STATUS_FILE"
  echo "status=complete" >>"$STATUS_FILE"
else
  echo "status=failed_missing_zip" >>"$STATUS_FILE"
  exit 1
fi
