# Detector training I/O fixes

Recorded 2026-08-06. Source: operator fix note for Zarr-bound U-Net training on HPC.

## Problem

GPU job 4556 (~14h+) was I/O-bound: ~133k GT nodes/epoch, random global shuffle,
full-frame Zarr reads, ~1.5/11 GB GPU memory. Checkpoint only after all epochs.

## Fix 1 — frame-grouped batch sampler (highest value)

Group training indices by `(dataset_name, t)`, shuffle frames each epoch, shuffle
cells inside each frame, emit batches from a single frame only. Replace
`shuffle=True` + `batch_size=` with `batch_sampler=`.

Recommended: `batch_size: 8`, `num_workers: 4`, `frame_cache_size: 2` (do **not**
raise frame cache to 32–64 with global shuffle — low hit rate, high RAM).

Expected: ~7× fewer full-frame reads; overall ~2–5× faster wall clock.

## Fix 2 — keep Zarr arrays open per worker

Per-process `_open_cache` on `VolumeDatasetReader` so each DataLoader worker does
not reopen stores / walk arrays repeatedly.

## Fix 3 — empty crops prefer anchor frame

`_empty_sample(dataset_name=..., t=anchor.t, ...)` tries the anchor frame first;
random frame only as fallback. Preserves frame locality.

## Fix 4 — disable CPU blur/noise for turnaround run

`noise_std: 0.0`, `blur_sigma_px: [0.0, 0.0]`. Keep flips/rot90/intensity.
Later: move noise/blur into the GPU training loop.

## Fix 5 — direct patch reads + precomputed frame percentiles (longer term)

Precompute `(dataset, t) → (lo, hi)` percentiles; `read_patch` + normalize patch
only. Requires inspecting real chunk shapes first. If chunks already are whole
frames, prefer Fix 1 full-frame grouping.

### Chunk inspection (2026-08-06, login01)

```text
All 199 train stores: shape (*, 64, 256, 256), chunks (1, 64, 256, 256)
```

Every Zarr chunk is an entire frame. Direct patch reads would still decompress the
full frame chunk, so Fix 5 is deferred; frame-grouped full-frame loading (Fix 1)
is the correct design for this dataset.

## Trainer ops (before next long run)

- `last.pt` every epoch; `best.pt` on validation improvement
- batch progress every 100–500 steps with elapsed time and samples/s
- automatic mixed precision
- resume support

## Next-run recipe

```yaml
epochs: 3
batch_size: 8
num_workers: 4
frame_cache_size: 2
augmentation:
  enabled: true
  flip_prob: 0.5
  rot90_prob: 0.5
  intensity_scale: [0.9, 1.1]
  intensity_shift: [-0.05, 0.05]
  noise_std: 0.0
  blur_sigma_px: [0.0, 0.0]
```

Plus: frame-grouped sampling, persistent Zarr handles, anchor-frame empty
sampling, per-epoch checkpointing.

## Job 4556

Do **not** cancel unless the remaining wall-clock budget is clearly insufficient.
No intermediate checkpoints exist; cancel discards all progress. Staging to
`/tmp/${USER}` is already correct; NFS is not the missing fix.
