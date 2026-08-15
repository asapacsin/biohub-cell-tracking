# Postprocess stage attribution for `postprocessing_removed`

Date: 2026-08-09  
Recipe: α=0.5, det=0.96875, safe-div ON, short-track ON (production).  
Method: replay `filter_output_graph` on saved raw GEFFs with per-stage edge checkpoints  
(`src/biohub_pipeline/postprocess_stage_trace.py`).

Watched edges = ordinary GT associations labeled `postprocessing_removed` in the
candidate-edge bottleneck diagnostics (ILP-selected, absent from final).

## Stage counts (first loss)

| split | n | motion_relink | short_track_filter | other |
|-------|--:|-------------:|-------------------:|------:|
| fixed-8 | 57 | **46 (80.7%)** | 11 (19.3%) | 0 |
| holdout-8 | 23 | **13 (56.5%)** | 10 (43.5%) | 0 |

No watched edge was first lost at distance filter, single-parent repair,
single-child repair, gap-close, safe-div, division geometry, or prune-isolated.

## Interpretation

1. **Short-track** accounts for a minority of these labels and was already shown
   (short-track ablation) to be **net helpful** — do not disable.
2. **Motion-relink** is the dominant *harmful-looking* removal stage among
   bottleneck `postprocessing_removed` edges. It replaces the ILP edge set with
   a motion+learned-prob matching; true ILP edges can disappear even when both
   endpoints survive.
3. Single-parent repair is **not** the first-loss stage for these watched edges
   (it may still matter for FP control).

## Next experiment

Ablate `output_motion_relink=false` (and single-parent OFF as a secondary check)
postprocess-only on the same GEFFs. Promote only if fixed-8 and holdout both
improve (or fixed-8 clear gain without holdout regression).

Artifacts: `outputs/experiments/postprocess_stage_attribution_v1/`
