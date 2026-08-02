# Durable project memory

## Sources of truth

- The local `sample_submission.csv`, test `.zarr` stores, training images, annotations, and Zarr
  metadata override prompt assumptions.
- As of 2026-08-02, those authoritative files were not found in the workspace or common local
  download/data directories. No array shapes, axes, spacing, annotation schema, or sample dtypes
  have been inferred.

## Design decisions

- Competition identity is a per-detection `node_id`, allocated per dataset; biological continuity
  and divisions are represented only by directed edges.
- Submission voxel coordinates are always integer `(z, y, x)`; tracking distances will use
  physical `(z, y, x)` values derived from dataset metadata.
- Core modules are storage-agnostic. Only a local storage backend exists in Milestone 1.
- Detector and tracker implementation are deliberately deferred until official-data inspection
  succeeds.
- Agent handoff files never supersede official competition files and must not contain secrets or
  private raw data.

## Current handoff

- Run `biohub-track inspect --competition-root data/competition` after adding the Kaggle download.
- Commit the generated inspection report or summarize verified facts here before starting
  Milestone 2.

## Verified Milestone 1 foundation

- On 2026-08-02, the reader and inspection report were extended to inspect both test and training
  Zarr metadata while preserving metadata-first, frame-lazy access.
- The deterministic fixture now contains a test Zarr store, training Zarr store, tracking table,
  and sample submission. Its `inspect`, `validate-data`, and `validate-submission` CLI paths pass.
- The submission validator requires every numeric output column to be NumPy `int64`.
- Verification passed with 30 pytest tests, Ruff lint/format checks, and strict mypy.
- Official files were not found in the workspace, Administrator Downloads/Desktop,
  `D:\Downloads`, or `D:\桌面`; authoritative Milestone 0 findings remain blocked.
