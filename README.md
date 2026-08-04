# Biohub – Cell Tracking During Development

Local-first learned cell-detection and lineage-tracking architecture for the Biohub competition.
It combines anisotropy-aware 3D detection, sparse temporal candidates, replaceable association
scoring, globally constrained lineage optimization, and strict node/edge submission validation.

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the complete target design and package
contracts. The validated classical public-test baseline remains available as a reproducible fallback
while learned model artifacts are trained.

## Public-test baseline

Process all 4 public test Zarr videos, save detections/tracks/visualizations, and write a validated
submission:

```bash
.venv/bin/python scripts/run_public_test_baseline.py \
  --input-dir data/competition \
  --output-dir outputs/public_test_baseline \
  --config configs/baseline.yaml
```

Optional: `--video-id NAME`, `--start-frame N`, `--end-frame N`, `--no-save-visualizations`.

Outputs:

```text
outputs/public_test_baseline/
  detections/<video>.csv
  tracks/<video>.csv
  visualizations/<video>/frame_XXXX.png   # MIP overlays with cell_id
  diagnostics/<video>.json
  submission/submission.csv
  baseline_report.json
```

Internal tracking uses persistent `cell_id` values. The competition export uses per-detection
`node_id` rows plus continuation/lineage `edge` rows (see sample submission). Division children
store `parent_id` (= parent `cell_id`) and export as two outgoing edges from the parent node.
Toggle via `tracking.division_enabled` in `configs/baseline.yaml`.

## Modular inference pipeline

The primary `biohub-track run` path composes metadata validation, physical-unit preprocessing,
learned heatmap ensembling (or the blob fallback), adaptive decoding, sparse link/gap/division
candidates, replaceable scoring, global ILP selection, graph cleanup, and strict submission export.

Use `configs/local_baseline.yaml` for a dependency-light smoke test. The learned configuration is
`configs/architecture.yaml`; update its explicit model artifact paths before running it.

## Current authoritative-data status

Official volumes are Zarr v3 with axes `(t,z,y,x)` and spacing `(1.625, 0.40625, 0.40625)` µm.
Training labels are `.geff` graphs. Place the Kaggle download under `data/competition`.

Expected inputs are discovered, not assumed:

- exactly one `sample_submission.csv` anywhere below the competition root;
- test stores below `test/**/*.zarr`;
- training image stores below `train/**/*.zarr`, with their metadata inspected separately;
- table-like training annotations discovered by name/location.

The reader accepts axis definitions only from OME-NGFF `multiscales.axes`, `_ARRAY_DIMENSIONS`, or
Zarr `dimension_names`. It requires coordinate scale metadata and refuses to guess either axes or
voxel spacing.

## Setup

Python 3.12 is the primary target.

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

Detector training additionally requires the ML extra:

```powershell
python -m pip install -e ".[dev,ml]"
```

## Commands

```powershell
biohub-track inspect --competition-root data/competition
biohub-track validate-data --competition-root data/competition
biohub-track validate-submission `
  --submission outputs/baseline/submission.csv `
  --competition-root data/competition
biohub-track train-detector-ensemble --competition-root data/competition --config configs/training.yaml
biohub-track train-association --competition-root data/competition --config configs/training.yaml
biohub-track run --competition-root data/competition --config configs/architecture.yaml
```

`inspect` reads group/array metadata and small table samples but never loads a full image volume. It
writes `outputs/inspection_report.json`. `validate-data` fails if an authoritative input is missing
or its metadata cannot be interpreted without guessing.

`train-detector` fits one 3D U-Net; `train-detector-ensemble` fits every configured seed. Each writes
a checkpoint, TorchScript heatmap predictor, and JSON training manifest. `train-association` fits the
sparse-event scorer and writes its portable JSON artifact. `run` loads those artifacts and executes
the globally optimized inference pipeline.

## Competition graph contract

Every detection at every time point is its own node. Node IDs are unique within a dataset and may
restart in the next dataset. Continuation and lineage relationships are directed edges; a division
is one source node with two outgoing edges. A target node never inherits its source node ID.

Submission columns are fixed in this order:

```text
id,dataset,row_type,node_id,t,z,y,x,source_id,target_id
```

Node coordinates are integer voxel coordinates in `(z, y, x)`. Physical-space calculations use
`voxel_spacing_zyx` read from that dataset's metadata. The strict validator checks schema, integer
dtypes, sentinels, row IDs, dataset names, node/edge integrity, bounds, time, single parentage,
forward-time edges, duplicates, and cycles. It also compares the generated schema with the local
sample submission.

## Synthetic fixture

The deterministic fixture is for infrastructure tests only; it is not evidence that detection or
tracking works.

```powershell
python scripts/generate_tiny_fixture.py data/sample
biohub-track inspect --competition-root data/sample
biohub-track validate-data --competition-root data/sample
```

It contains nodes 1–7 and edges `1→3`, `2→4`, `3→5`, `3→6`, `4→7`, including a division at node 3.

## Python API

```python
from biohub_tracker.submission import build_submission, validate_submission

graphs = run_prediction_pipeline(competition_root, config)
submission = build_submission(graphs)
validate_submission(submission, competition_root)
```

## Repository map

```text
configs/                     hypothesis-only runtime settings
data/competition/            ignored authoritative Kaggle files
data/sample/                 optional deterministic fixture
notebooks/                   guarded future Kaggle adapter
scripts/                     inspection and fixture entry points
src/biohub_tracker/
  annotation_reader.py       table discovery and schema samples
  association/               sparse candidates, scoring, global optimization
  coordinates.py             centralized voxel/physical conversions
  detection/                 blob fallback and learned heatmap inference/decoding
  inspection.py              competition discovery and JSON reporting
  models.py                  typed data and graph models
  preprocessing.py           metadata checks and robust intensity normalization
  postprocessing.py          graph cleanup and optional short-track filtering
  storage.py                 local storage abstraction
  zarr_reader.py             lazy metadata-first frame reader
  submission/                exact writer and strict validator
  training/                  Zarr/GEFF datasets, targets, labels, and model trainers
tests/                       unit and integration tests
.agents/                     persistent handoff state and durable memory
```

## Agent handoff memory

Future agents must read `AGENTS.md`, `.agents/state.json`, and `.agents/memory.md`. These files store
verified progress, blockers, decisions, and next actions. They never replace official files, contain
raw private data, or silently promote guesses to facts.

## Next milestone gate

Run cross-validation on the official training stores, retain immutable artifact manifests, and
compare the learned submission against the validated public-test baseline. Numerical values in the
YAML files remain starting hypotheses until that evaluation is recorded.
