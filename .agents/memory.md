# Durable project memory

## Modular learned architecture (2026-08-04)

- The primary `pipeline.run_prediction_pipeline` composition is now detector -> sparse temporal
  candidates -> replaceable scorer -> global optimizer -> post-processing -> submission graph.
- Learned detection uses the framework-neutral `HeatmapPredictor` contract; the included runtime
  adapter loads TorchScript heatmap models and supports model/seed ensembling plus axis-flip TTA.
- The heatmap decoder performs adaptive thresholding, physical-unit anisotropic 3D NMS, and local
  subvoxel refinement. The classical blob detector remains a selectable fallback and enters the
  same downstream graph pipeline.
- Sparse association candidates include one-frame links, configurable gap links, and atomic paired
  division events. Edge features carry geometry/motion/confidence plus optional appearance,
  intensity, volume, and temporal-density context.
- The included reference scorer is deterministic and replaceable through `CandidateGraphScorer`.
  The concrete learned scorer is a class-balanced logistic event model with a portable JSON
  artifact; richer Trackastra/HOCT-style adapters can use the same scorer contract.
- The ILP optimizer enforces at most one incoming parent and at most one outgoing event per node;
  division events produce two edges atomically. It falls back to a constraint-aware greedy solver
  if MILP fails or is explicitly selected.
- Training now includes same-stem Zarr/GEFF pairing, lazy positive-centred 3D patches, physical-unit
  Gaussian targets, an encoder-decoder 3D U-Net, BCE-plus-Dice fitting, dataset-level validation
  splits, best-checkpoint retention, TorchScript export, and multi-seed ensemble commands.
- Learned full-frame inference uses overlapping patch windows before model/seed and flip-TTA
  averaging, so patch-trained U-Nets are not applied to the entire volume in one allocation.
- Sparse candidate labels still use `-1` for detections without an explicit GEFF-node match. The
  learned association trainer fits known edge and atomic-division event labels only.
- Target design: `docs/ARCHITECTURE.md`; training config: `configs/training.yaml`; learned inference
  config: `configs/architecture.yaml`.
- Windows verification on 2026-08-04: 58 pytest tests passed using a repository-local basetemp;
  Ruff passed; strict mypy passed for 42 source files. The local basetemp avoids an unrelated access
  denial in `%TEMP%/pytest-of-Administrator`.
- This Windows checkout had neither PyTorch nor `data/competition/train` on 2026-08-04. The model
  training code and artifact paths were verified structurally and with data/artifact unit tests,
  but full GPU fitting must run on the host that holds the official stores. Earlier memory saying
  the full dataset is under `data/competition` refers to the previously inspected host, not this
  checkout.

## Sources of truth

- The local `sample_submission.csv`, test `.zarr` stores, training images, annotations, and Zarr
  metadata override prompt assumptions.
- Competition slug: `biohub-cell-tracking-during-development`. Credentials live in
  gitignored `.kaggle/kaggle.json` (use `KAGGLE_CONFIG_DIR` / `KAGGLE_API_TOKEN`).
- Local official inspection (2026-08-02) of `data/competition` sample `44b6_0113de3b`:
  - Image Zarr v3 group with OME-NGFF multiscales axes `T,Z,Y,X` (normalized to `t,z,y,x`).
  - Array `0` shape `(100, 64, 256, 256)` `uint16`, chunks `(1, 64, 256, 256)`,
    voxel spacing_zyx `(1.625, 0.40625, 0.40625)`.
  - Training annotations are `.geff` Zarr v3 graphs (not CSV): `nodes/ids`,
    `nodes/props/{t,z,y,x}/values`, `edges/ids` shape `(N,2)`; sparse labels;
    `attributes.geff.extra.estimated_number_of_nodes` (25755 for this sample; 52 labeled nodes /
    50 edges observed).
  - `sample_submission.csv` columns/dtypes match the project contract (`int64` numerics).
- Full dataset unzipped under `data/competition` (zip removed after extract).
- Layout verified: 4 test `.zarr`, 199 train `.zarr`, 199 train `.geff`.
- `validate-data` and sample-schema `validate-submission` pass on this host.

## Design decisions

- Competition identity is a per-detection `node_id`, allocated per dataset; biological continuity
  and divisions are represented only by directed edges.
- Submission voxel coordinates are always integer `(z, y, x)`; tracking distances will use
  physical `(z, y, x)` values derived from dataset metadata.
- Core modules are storage-agnostic. Only a local storage backend exists in Milestone 1.
- Detector uses anisotropic Gaussian sigma / separation in µm converted by voxel spacing.
- Division heuristic is implemented as a greedy scored post-pass after one-to-one linking;
  volume consistency is optional (`volume_weight`, off by default).
- Agent handoff files never supersede official competition files and must not contain secrets or
  private raw data.
- Cursor project rules live under `.cursor/rules/` and summarize the same handoff contract for the
  IDE agent; they do not replace `.agents/state.json` / `.agents/memory.md`.

## Current handoff

- Public-test baseline:
  `scripts/run_public_test_baseline.py --input-dir data/competition --output-dir outputs/public_test_baseline --config configs/baseline.yaml`
- Division post-pass is wired end-to-end (`DivisionConfig` from YAML → tracker →
  `observations_to_graph` parent→daughter edges; tracks CSV includes `division_score`).
- `configs/baseline.yaml` has `division_enabled: true` with GEFF-tuned knobs
  (`require_matched_daughter`, min/max daughter separation, mid-point cap, ≤1 div/frame).
- Validated tuned submission: **38077 nodes + 32917 edges**; **302** divisions
  (all branching parents out-degree exactly 2). Earlier over-accept: 3278 → 569 → 302.
- Train GEFF labeled rate is ~0.8 divisions/video (sparse labels); 302 across 4 test videos
  is still a geometric heuristic upper bound, not GT.
- Artifacts under `outputs/public_test_baseline/`.
- Next: optional Kaggle submit or GEFF precision audit.
- Linux `.venv` is ready; use `.venv/bin/python` / `.venv/bin/biohub-track`.
- Cursor Cloud / My Machines is a poor fit for campus-only HPC; use local Agent on `login01`.

## Verified Milestone 1 foundation

- On 2026-08-02, the reader and inspection report were extended to inspect both test and training
  Zarr metadata while preserving metadata-first, frame-lazy access.
- The deterministic fixture now contains a test Zarr store, training Zarr store, tracking table,
  and sample submission. Its `inspect`, `validate-data`, and `validate-submission` CLI paths pass.
- The submission validator requires every numeric output column to be NumPy `int64`.
- Verification passed with 30 pytest tests, Ruff lint/format checks, and strict mypy (Windows).
- Official files were not found in the workspace, Administrator Downloads/Desktop,
  `D:\Downloads`, or `D:\桌面`; authoritative Milestone 0 findings remain blocked.
- Linux host (2026-08-02): `python3.12-venv` apt package needs sudo (unavailable). Created
  `.venv` via `python3.12 -m venv --without-pip` + `get-pip.py`, then
  `pip install -e ".[dev]"`. Regenerated `data/sample` fixture. Fresh verification: 30 pytest,
  ruff check/format, mypy, and fixture CLI inspect/validate-data/validate-submission all passed.
  `biohub-track inspect --competition-root data/competition` still reports all authoritative
  inputs missing.

## Project shape (durable)

- Package: `biohub_tracker` (`src/biohub_tracker/`), CLI entry `biohub-track`.
- Milestone 0: discover/inspect official competition layout without loading full volumes.
- Milestone 1: typed models, Zarr metadata reader, annotation discovery, submission writer +
  strict validator, deterministic tiny fixture, guarded `run` pipeline.
- Milestone 2+: classical detector (`detection/`), then tracker (`tracking/`); `pipeline.run_prediction_pipeline` currently raises `NotImplementedError`.
- Submission columns (fixed order): `id,dataset,row_type,node_id,t,z,y,x,source_id,target_id`.
- Nodes are per-detection; edges encode continuation/lineage; division = one parent, two outgoing edges; target never inherits source `node_id`.
- YAML config numeric values are unconfirmed hypotheses until tuned on real training data.
