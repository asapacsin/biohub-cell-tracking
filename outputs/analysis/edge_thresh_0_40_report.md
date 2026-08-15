# Edge threshold 0.40 under motion-relink OFF

Date: 2026-08-10  
Control: promoted edge=0.45 motion-OFF (fixed-8 0.915193 / holdout 0.962739).

## Results

| set | control (0.45) | edge 0.40 | Δ | edges TP/FP/FN |
|-----|---------------:|----------:|--:|----------------|
| fixed-8 | 0.915193 | **0.918144** | **+0.002951** | 3949/187/154 |
| holdout-8 | 0.962739 | **0.964673** | **+0.001933** | 4037/76/89 |

## Conclusion

**PROMOTE edge_threshold=0.40.** Smaller gains than 0.5→0.45; stop further
threshold lowering without a fresh error taxonomy.

Promoted config: `configs/experiments/recipe_c_motion_off_edge_0_40_det0_96875.yaml`  
Artifacts: `outputs/experiments/edge_thresh_0_40_motion_off_v1/`
