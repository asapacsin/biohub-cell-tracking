# Short-track filter ablation (postprocess-only)

Date: 2026-08-09  
Recipe held fixed: α=0.5, det=0.96875, safe-div ON, gap2 OFF, DeepCenter OFF.  
Raw GEFFs reused (no re-inference):  
- fixed-8: `candidate_edge_bottleneck_v1/raw_geff`  
- holdout-8: `holdout8/det0_96875_safeon/raw_geff`

Code: `filter_short_track_components` in `src/biohub_pipeline/postprocessing.py`  
(control: `output_filter_short_tracks=true`, `output_min_track_len=6`, adaptive rescue ON).

## Control reproduction

| set | score | Δ |
|-----|------:|--:|
| fixed-8 | **0.884746427159** | 0 (exact) |
| holdout-8 | **0.959013051622** | 0 (exact) |

Control removes 6637 / 6485 short-track nodes on fixed-8 / holdout-8.

## Results

| variant | fixed-8 score | Δ fixed-8 | holdout score | Δ holdout | nodes removed (f8) |
|---------|--------------:|----------:|--------------:|----------:|-------------------:|
| control (min_len=6, rescue ON) | **0.884746** | 0 | **0.959013** | 0 | 6637 |
| min_len=4, rescue OFF | 0.883637 | −0.001109 | 0.957024 | −0.001989 | 1526 |
| min_len=3, rescue OFF | 0.881681 | −0.003065 | 0.957678 | −0.001335 | 116 |
| min_len=2, rescue OFF | 0.881649 | −0.003097 | 0.957854 | −0.001159 | 0 |
| filter OFF | 0.881649 | −0.003097 | 0.957854 | −0.001159 | 0 |

## Interpretation

Retaining short tracks **hurts** the official-spec score. Every relaxation of the
short-track filter is worse on both fixed-8 and holdout.

- Disabling the filter adds FP more than it recovers FN (fixed-8 edges 3878/283/225 → 3883/290/220).
- Per-dataset, OFF helps `44b6_0113de3b` (+0.037) but regresses most `6bba_*` sets
  (largest: `6bba_fc83837d` −0.007).
- `min_len=2` matches filter OFF: only isolated nodes would be removed, and that is
  score-equivalent to keeping everything.

So the bottleneck label `postprocessing_removed` is **not** a mandate to disable
short-track cleanup. Short-track removal is net-positive FP control; many removed
components are not recoverable true tracks under this metric.

## Decision

**KEEP CONTROL short-track settings.** Do not promote OFF or lower `min_track_len`.

## Best next step

Stay postprocess-only on saved GEFFs, but stop ablating short-track.

Attribute the `postprocessing_removed` causal errors from the bottleneck run by
**stage** (short-track vs earlier repairs: gap-close / motion-relink / safe-div /
single-parent repair). If non-short-track stages dominate harmful removals, ablate
that one stage next. If short-track accounts for most `postprocessing_removed`
labels, treat those as beneficial cleanup and shift focus back to
association-ranking / ILP with the production short-track filter left ON.

Artifacts: `outputs/experiments/shorttrack_ablation_det0_96875/comparison.json`
