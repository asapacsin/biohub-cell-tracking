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

## Kaggle code-competition submit (2026-08-04/05)

- This competition is **notebook-only**. Internet must be **Off** for Submit.
- Competition data mount:
  `/kaggle/input/competitions/biohub-cell-tracking-during-development` (`test/*.zarr`).
- User code dataset:
  `/kaggle/input/datasets/asapacsinfarland/biohub-code/biohub-cell-tracking`
  (do **not** point `COMP_ROOT` at `data/sample` / `tiny.zarr` inside that zip).
- Offline package installs: Kaggle cold start often lacks `zarr` / sometimes `numcodecs`.
  - Pure wheels dataset: `/kaggle/input/datasets/asapacsinfarland/wheels`
    (`donfig`, `packaging`, `typing_extensions`, `zarr-*-py3-none-any.whl`).
  - Linux wheels dataset: `/kaggle/input/datasets/asapacsinfarland/linux-wheel`
    (`numcodecs-*-manylinux*.whl`, `google_crc32c` / `crc32c` manylinux).
  - Install with `pip install --no-deps <exact.whl>`; never `--find-links` the Win
    `win_amd64` wheels in the pure folder (pip may try them and fail).
- Warm editor sessions can show `import zarr` succeeding without install; **Submit is cold**.
  Always verify after Restart + Internet Off.
- Node rows correctly use `source_id=target_id=-1`; edge rows carry real IDs. Sorting puts
  all nodes before edges — previewing the CSV head is misleading.
- Proven Kaggle path: offline install → `run_public_test_baseline(COMP_ROOT, ...)` →
  copy to `/kaggle/working/submission.csv`. Expected classical counts ~38077 nodes / 32917 edges.
- Cursor Cloud / My Machines is a poor fit for campus-only HPC; work on `login01` or Kaggle.

## Learned pipeline gaps (2026-08-05)

- Authoritative need-to-do note: `docs/NEED_TO_DO.md`.
- Unit tests ≠ detector accuracy; no recorded detector CV or learned LB yet.
- Priority order: (1) video-fold detector metrics at 2/4/6 µm,
  (2) data pipeline, (3) anisotropic vs isotropic U-Net ablations, (4) association
  from OOF detections not perfect GEFF, (5) node rejection after calibration.
- Data-pipeline hardening landed 2026-08-05 in `CentroidPatchDataset`:
  LRU frame cache, `nodes_by_time`, `set_epoch` stochasticity, patch mix
  80% positive / 15% near-miss / 5% empty, train-only XY flip/rot90 + photometric
  aug, `DatasetView` for eval-safe validation. Config: `configs/training.yaml`.
  Still open from that step: AMP and resume/checkpoints; FP mining later.
- Known remaining gaps: isotropic Conv3d/MaxPool on anisotropic voxels, association
  features zeroed under GEFF-centroid training, ILP selects edges not nodes,
  dense-frame percentile threshold + plateau NMS duplicates.
- `artifacts/association/model.json` exists from GEFF-only training (14 features);
  treat as geometric baseline only until OOF retrain.

## Current handoff

- Host `login01` has official `data/competition/test` (4 zarrs) and local validated
  `outputs/public_test_baseline/submission/submission.csv` (~38077 nodes / 32917 edges / 302 divisions).
- Architecture modular pipeline now has validated public-test submissions without U-Net:
  - blob + handcrafted: `outputs/architecture_blob_ilp/submission.csv` (37266 / 34101 / 2)
  - blob + learned linear: `outputs/architecture_blob_learned/submission.csv` (35771 / 30833 / 0)
  - Summary: `outputs/ARCHITECTURE_RESULTS.md`
- Optimizer: sparse COO incidence matrix; greedy fallback when events > `ilp_event_limit` (40k).
- Architecture configs drop isolated tracks (`minimum_track_length=2`).
- Classical baseline still has the tuned division post-pass (302 divisions); architecture
  atomic-division path is under-firing and needs calibration / OOF association.
- Classical baseline command:
  `scripts/run_public_test_baseline.py --input-dir data/competition --output-dir outputs/public_test_baseline --config configs/baseline.yaml`
- Full GPU fitting still needs a host with PyTorch + train stores; follow `docs/NEED_TO_DO.md`.
- Linux `.venv` on login01: `.venv/bin/python` / `.venv/bin/biohub-track`.

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
- Milestone 2+: classical detector + greedy NN + division baseline; learned detector/association/ILP path also present.
- Submission columns (fixed order): `id,dataset,row_type,node_id,t,z,y,x,source_id,target_id`.
- Nodes are per-detection; edges encode continuation/lineage; division = one parent, two outgoing edges; target never inherits source `node_id`.
- YAML config numeric values remain hypotheses until scored / GEFF-validated.

## Detector training I/O fixes (2026-08-06)

- Root cause of slow GPU train: full-frame Zarr I/O under global shuffle (~133k
  samples/epoch), not GPU compute (~1.5 GB VRAM used).
- All 199 train Zarrs use chunks `(1, 64, 256, 256)` (whole frame). Direct patch
  reads (Fix 5) would not reduce decompression; deferred.
- Implemented: `FrameGroupedBatchSampler`, per-process `VolumeDatasetReader._open_cache`,
  anchor-frame-first empty crops, disabled CPU noise/blur for turnaround runs,
  per-epoch `*_last.pt` / `*_best.pt`, AMP, resume, batch progress logs.
- Spec/note: `docs/TRAINING_IO_FIXES.md`. Next-run defaults in `configs/training.yaml`
  and `scripts/slurm/remote_train_only.sh`: epochs=3, batch_size=8, frame_cache_size=2,
  frame_grouped_batches=true.
- Job 4556 (pre-fix code) left running; cancel would lose all progress (no mid-epoch ckpt).
