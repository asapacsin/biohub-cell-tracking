# Remaining `postprocessing_removed` under motion-relink OFF

Date: 2026-08-10  
Config: recipe C + `output_motion_relink=false`, det=0.96875, short-track ON.  
Watched edges: ordinary associations labeled `postprocessing_removed` after rediagnosis
on motion-OFF finals.

## Stage counts (first loss)

| split | n | short_track_filter | other |
|-------|--:|-------------------:|------:|
| fixed-8 | 23 | **23 (100%)** | 0 |
| holdout-8 | 14 | **14 (100%)** | 0 |

## Interpretation

With motion-relink disabled, every remaining bottleneck `postprocessing_removed`
edge is removed by short-track filtering. Short-track ablation (E1) already showed
that disabling/relaxing that filter hurts score. Treat these remaining labels as
**beneficial FP cleanup**, not a retention bug.

Artifacts: `outputs/experiments/postprocess_stage_attribution_motion_off_v1/`
