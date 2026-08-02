# Biohub – Cell Tracking During Development

Local-first, cloud-ready infrastructure for inspecting the official competition data and producing
strict node-and-edge Kaggle submissions. This repository currently implements **Milestone 0** and
**Milestone 1** only. Detection, tracking, and division inference are intentionally not yet
implemented.

## Current authoritative-data status

Inspection on 2026-08-02 found no downloaded competition files in this workspace or the common
local download/data folders that were checked. Consequently, this project does **not** claim an
actual Zarr shape, axis order, voxel spacing, annotation schema, or sample-submission dtype yet.
Place the unmodified Kaggle download under `data/competition` and run the inspection commands below.

Expected inputs are discovered, not assumed:

- exactly one `sample_submission.csv` anywhere below the competition root;
- test stores below `test/**/*.zarr`;
- training image stores below `train/**/*.zarr`;
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

## Commands

```powershell
biohub-track inspect --competition-root data/competition
biohub-track validate-data --competition-root data/competition
biohub-track validate-submission `
  --submission outputs/baseline/submission.csv `
  --competition-root data/competition
```

`inspect` reads group/array metadata and small table samples but never loads a full image volume. It
writes `outputs/inspection_report.json`. `validate-data` fails if an authoritative input is missing
or its metadata cannot be interpreted without guessing.

The `run` command is present as a stable interface but raises an explicit `NotImplementedError` in
Milestones 0–1. That guard prevents an empty or invented tracker from being mistaken for a working
baseline.

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
```

It contains nodes 1–7 and edges `1→3`, `2→4`, `3→5`, `3→6`, `4→7`, including a division at node 3.

## Python API

```python
from biohub_tracker.submission import build_submission, validate_submission

# Available after later milestones implement the guarded pipeline:
# graphs = run_prediction_pipeline(competition_root, config)
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
  coordinates.py             centralized voxel/physical conversions
  inspection.py              competition discovery and JSON reporting
  models.py                  typed data and graph models
  storage.py                 local storage abstraction
  zarr_reader.py             lazy metadata-first frame reader
  submission/                exact writer and strict validator
tests/                       Milestone 0–1 unit/integration tests
.agents/                     persistent handoff state and durable memory
```

## Agent handoff memory

Future agents must read `AGENTS.md`, `.agents/state.json`, and `.agents/memory.md`. These files store
verified progress, blockers, decisions, and next actions. They never replace official files, contain
raw private data, or silently promote guesses to facts.

## Next milestone gate

Do not begin the classical detector until `validate-data` passes on the official download and the
resulting report has been reviewed. All numerical values in the YAML files are unconfirmed starting
hypotheses and must be tuned on training data.

