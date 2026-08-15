# Edge threshold 0.45 under motion-relink OFF

Date: 2026-08-10  
Prior: motion-relink OFF promoted (fixed-8 0.906725 / holdout 0.960307).  
Rediagnosis showed 48/16 ordinary associations blocked by edge_threshold=0.5
with blended probs mostly in [0.39, 0.50).

## Experiment

- **Change:** `inference.edge_threshold=0.45` via new optional CLI patch;
  postprocess keeps `output_motion_relink=false`.
- **Compute:** full two-seed GPU inference, fixed-8 + holdout-8.
- **Control:** motion-OFF, edge=0.5 (E3 scores).

## Results

| set | control (0.5) | edge 0.45 | Δ | edges TP/FP/FN |
|-----|--------------:|----------:|--:|----------------|
| fixed-8 | 0.906725 | **0.915193** | **+0.008468** | 3927/185/176 (was 3889/195/214) |
| holdout-8 | 0.960307 | **0.962739** | **+0.002432** | 4031/82/95 (was 4017/84/109) |

## Conclusion

**PROMOTE edge_threshold=0.45.** Both sets improve; TP↑ and FN↓ with fewer FP on both.

Promoted combined config:
`configs/experiments/recipe_c_motion_off_edge_0_45_det0_96875.yaml`

Artifacts: `outputs/experiments/edge_thresh_0_45_motion_off_v1/`
