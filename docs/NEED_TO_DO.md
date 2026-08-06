# Need to do — learned pipeline gaps

Captured 2026-08-05. Numerical settings remain hypotheses until cross-validation
is performed. Unit tests (shapes, I/O, graph constraints, sliding-window
reconstruction, artifact loading) do **not** establish detection precision/recall
or tracking accuracy.

## Main problems

### 1. U-Net not proven on real training data

Largest issue is absence of recorded detector cross-validation and learned
submission results. A GPU host is still required for full learned fitting.

- 58 unit tests passing ≠ model detects cells accurately
- Need detection metrics on held-out complete videos, not only BCE/Dice

### 2. Training receives almost no data variation — DONE (2026-08-05)

Was: deterministic `default_rng(seed + index)` every epoch; positives only.

**Implemented in `CentroidPatchDataset`:**

- `set_epoch(epoch)` + SeedSequence RNG so samples vary across epochs
- Patch mix **80% positive / 15% near-miss / 5% empty** (`PatchMixConfig`)
- Near-miss: center clear of `positive_center_radius_um`; source preferred in
  40-70% XY annulus; all-zero target
- Empty: no centroid in crop ± `empty_exclusion_margin_um`; all-zero target
- Train-only XY flips / rot90 + intensity scale/shift, noise, blur
- Eval / `DatasetView(..., train=False)`: positives only, deterministic

Later: mine false-positive patches from model predictions (more valuable than
random empty backgrounds). Elastic deform still deferred.

### 3. Data loader wastes CPU / GPU time — DONE (2026-08-05)

**Implemented:**

- LRU frame cache: `(dataset, t) → normalized frame` (`frame_cache_size`, default 4)
- `nodes_by_time`: `(dataset, t) → centroid list` (no per-getitem full graph scan)

Config knobs live in `configs/training.yaml` under `training.detector`.

### 4. U-Net pooling ignores voxel anisotropy

Gaussian targets respect physical spacing, but the U-Net uses isotropic
`Conv3d(kernel_size=3)` and `MaxPool3d(2)`. With spacing ≈ `(1.625, 0.406, 0.406)` µm,
a 3×3×3 kernel covers ~4.875 µm in Z vs ~1.219 µm in Y/X.

**Better early architecture**

| Level | Kernel | Pool |
|-------|--------|------|
| 1 | `(1, 3, 3)` | `(1, 2, 2)` |
| 2 | `(1, 3, 3)` | `(1, 2, 2)` |
| deeper | `(3, 3, 3)` | `(2, 2, 2)` |

Alternatively resample volumes toward isotropic spacing before training.
Current design can still work; probably not strongest for these images.

### 5. Association train–inference mismatch

Association is trained on perfect GEFF centroids (`score=1.0`), not detector
outputs. No real appearance embedding, intensity, or volume → many advertised
features are constant/zero during training:

- `confidence_mean`
- `appearance_similarity`
- `intensity_log_ratio`
- `volume_log_ratio`

Learned model is mostly geometry: distance, direction, time gap, density,
division geometry — not appearance-aware tracking.

**Fix — OOF detector predictions**

1. Train U-Net on videos A, B, C
2. Infer on held-out video D
3. Match predictions to GEFF; build pos/neg link examples
4. Repeat every fold
5. Fit association on these noisy OOF detections

Exposes misses, duplicates, localization error, and false positives.

### 6. Every detection becomes a submission node

ILP selects edges, not nodes. Node IDs are allocated for every decoded
detection before edge selection. Current config keeps isolated one-node tracks
(`minimum_track_length: 1`, `remove_isolated_short_tracks: false`) → every U-Net
FP becomes a submission node.

**Fix**

- Node-selection variables: confidence reward, isolated-node / birth / death
  penalties, edge consistency
- Simpler first step: drop low-confidence isolated components after optimization

### 7. Heatmap decoder can fail in dense frames

Effective threshold:

```text
max(fixed_threshold, heatmap_99.5_percentile)
```

Dense frames raise the percentile and can suppress weaker true cells.
NMS uses exact equality with a maximum-filter output; flat high-probability
plateaus can yield duplicate detections.

**Fix**

- Tune thresholds from validation data
- Genuine greedy physical-distance NMS or connected-component handling for plateaus

## Recommended order of work

### First: prove the detector alone

Use complete videos as folds. Report:

- Precision / Recall / F1 at 2, 4, and 6 µm
- Mean localization error
- Predicted vs true cell count
- False positives per frame

Do **not** evaluate only BCE/Dice.

### Second: improve the data pipeline — mostly DONE (2026-08-05)

Done: frame caching, `nodes_by_time`, stochastic aug, 80/15/5 pos/near-miss/empty.

Still open: mixed-precision training; training resume / checkpoints.

### Third: test anisotropic U-Net variants

Same folds and detection metrics for:

- A: current isotropic 3D U-Net
- B: anisotropic early layers
- C: isotropically resampled input

### Fourth: train tracking from OOF predictions

Do not train association solely from perfect GEFF centroids.

### Fifth: add node rejection

Only after detection and association are calibrated, rely on the global
optimizer for node selection.

## Related paths

- Architecture: `docs/ARCHITECTURE.md`
- Training config: `configs/training.yaml`
- Learned inference: `configs/architecture.yaml`
- Classical validated baseline (fallback LB path):
  `outputs/public_test_baseline/submission/submission.csv`
