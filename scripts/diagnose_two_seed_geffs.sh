#!/usr/bin/env bash
# Persist top-level raw_geff + convert saved two-seed GEFFs with outdegree diagnostics.
set -euo pipefail

DEST="${DEST:-/tmp/${USER}/biohub-twoseed-diag}"
OUT="${OUT:-${HOME}/biohub-outputs/fixed8/two_seed_alpha_0_5}"
RAW_SRC="${OUT}/predictions/raw_geff"
RAW_DST="${OUT}/raw_geff"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

cd "$DEST"
source "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate biohub
export PYTHONPATH="${DEST}/src"

mkdir -p "$RAW_DST"

python - <<PY
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

raw_src = Path("${RAW_SRC}")
raw_dst = Path("${RAW_DST}")
assert raw_src.is_dir(), raw_src
geffs = sorted(p for p in raw_src.iterdir() if p.name.endswith(".geff"))
assert len(geffs) == 8, geffs
for src in geffs:
    dest = raw_dst / src.name
    if dest.exists():
        if dest.is_dir():
            shutil.rmtree(dest)
        else:
            dest.unlink()
    shutil.copytree(src, dest)
manifest = {
    "schema_version": 1,
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "geff_count": len(geffs),
    "filenames": [p.name for p in geffs],
    "dataset_ids": [p.stem for p in geffs],
    "git_commit": "ce2e81a6cca0c8948764f9d82b6b1c9a187025fb",
    "config_path": "configs/clean_v106_two_seed.yaml",
    "note": "Copied from predictions/raw_geff persisted by failed two-seed job; no re-inference",
    "raw_geff_dir": str(raw_dst),
}
(raw_dst / "MANIFEST.json").write_text(
    json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
print("PERSISTED", len(geffs), raw_dst)
print("FILES", [p.name for p in geffs])
PY

echo "=== convert with diagnostics ==="
python - <<PY
from pathlib import Path
from biohub_pipeline.config import load_config
from biohub_pipeline.submission import write_submission_from_geff

out = Path("${OUT}")
geffs = sorted((out / "raw_geff").glob("*.geff"))
config = load_config(Path("configs/clean_v106_two_seed.yaml"))
pred_csv = out / "predictions" / "postprocessed_submission.csv"
try:
    summary = write_submission_from_geff(
        geffs,
        config,
        Path("data/competition/train"),
        pred_csv,
    )
    print("CONVERT_OK", summary)
except Exception as exc:
    print("CONVERT_FAIL", type(exc).__name__)
    print(str(exc))
    raise SystemExit(2)
PY

echo DIAG_DONE
ls -la "${RAW_DST}" | head
ls -la "${OUT}/predictions" | head
test -f "${OUT}/predictions/outdegree_violations.json" && echo "HAS_DIAGNOSTICS=1"
