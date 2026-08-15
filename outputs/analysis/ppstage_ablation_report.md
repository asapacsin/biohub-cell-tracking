# Postprocess stage ablation (motion-relink / single-parent)

Date: 2026-08-10  
Prior finding: stage attribution showed `motion_relink` first-loses **80.7%** of fixed-8
and **56.5%** of holdout `postprocessing_removed` bottleneck edges; short-track the rest.  
Method: postprocess-only on saved recipe-C det=0.96875 raw GEFFs (no re-inference).

## Control reproduction

| set | score | Δ |
|-----|------:|--:|
| fixed-8 | **0.884746427159** | 0 (exact) |
| holdout-8 | **0.959013051622** | 0 (exact) |

## Results

| variant | fixed-8 | Δ fixed-8 | holdout | Δ holdout | edges TP/FP/FN (fixed-8) |
|---------|--------:|----------:|--------:|----------:|--------------------------|
| control (motion ON, parent ON) | 0.884746 | 0 | 0.959013 | 0 | 3878/283/225 |
| **motion_relink OFF** | **0.906725** | **+0.021979** | **0.960307** | **+0.001294** | **3889/195/214** |
| single_parent OFF | 0.884746 | 0 | 0.959013 | 0 | identical |
| motion+parent OFF | 0.906725 | +0.021979 | 0.960307 | +0.001294 | identical to motion OFF |

## Interpretation

- Disabling motion-relink is a large fixed-8 win driven mainly by **FP collapse**
  (283 → 195) with modest TP/FN improvement.
- Holdout aggregate improves; gains are concentrated on `6bba_*` and `44b6_0db75fae`,
  with small regressions on three `44b6_*` sets. Net holdout still positive.
- Single-parent repair is score-noop on these GEFFs (with or without motion-relink).
- Hypothesis **supported**: motion-relink was the harmful postprocess rule behind most
  `postprocessing_removed` causal labels.

## Decision

**PROMOTE `output_motion_relink=false` for the local generalization-safe recipe.**  
Do **not** silently overwrite historical baseline YAMLs; use:

- `configs/experiments/recipe_c_motion_relink_off_det0_96875.yaml`
- alias: `configs/experiments/ppstage_motion_relink_off_det0_96875.yaml`

Old production reference (motion ON) remains for reproducibility comparisons.

Artifacts: `outputs/experiments/ppstage_ablation_det0_96875/comparison.json`
