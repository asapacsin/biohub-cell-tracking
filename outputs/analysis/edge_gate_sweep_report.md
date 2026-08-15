# Edge-gate retention ablation (motion-relink OFF base)

Date: 2026-08-12  
Control: motion_relink OFF, edge_threshold=**0.50**, α=0.5, det=0.96875  
  fixed-8 **0.906725** / holdout **0.960307**

## Sweep results (official-spec score)

| thr | fixed-8 | Δ vs 0.50 | edges TP/FP/FN | div TP/FP/FN | admitted | +adm | holdout | Δ vs 0.50 | edges TP/FP/FN | admitted | +adm |
|----:|--------:|----------:|----------------|-------------:|---------:|-----:|--------:|----------:|----------------|---------:|-----:|
| 0.50 | 0.906725 | 0 | 3889/195/214 | 0/16/7 | 167274 | 0 | 0.960307 | 0 | 4017/84/109 | 151219 | 0 |
| 0.45 | 0.915193 | +0.00847 | 3927/185/176 | 0/10/7 | 171679 | +4405 | 0.962739 | +0.00243 | 4031/82/95 | 154991 | +3772 |
| **0.40** | **0.918144** | **+0.01142** | **3949/187/154** | **0/10/7** | **174774** | **+7500** | **0.964673** | **+0.00437** | **4037/76/89** | **157720** | **+6501** |
| 0.35 | 0.919245 | +0.01252 | 3961/189/142 | 0/9/7 | 177048 | +9774 | 0.963895 | +0.00359 | 4041/80/85 | 160074 | +8855 |
| 0.30 | 0.917053 | +0.01033 | 3961/195/142 | 0/12/7 | 178869 | +11595 | 0.962579 | +0.00227 | 4041/84/85 | 161878 | +10659 |
| 0.25 | 0.915400 | +0.00868 | 3965/203/138 | 0/12/7 | 180469 | +13195 | 0.963224 | +0.00292 | 4044/82/82 | 163724 | +12505 |

vs 0.40: 0.35 fixed **+0.00110**, holdout **−0.00078** → reject as overfit.

## Probability-band diagnostic (capture top-k)

Fixed-8 precision_proxy (true ordinary / captured pairs):
- 0.45–0.50: 40/5358 = **0.75%**
- 0.40–0.45: 29/5299 = **0.55%**
- 0.35–0.40: 18/5706 = **0.32%**
- 0.30–0.35: 15/6428 = **0.23%**
- 0.25–0.30: 15/8008 = **0.19%**

Useful scorer signal exists in 0.40–0.50 (ILP recovers enough for net score gain).  
Below 0.40, bands are overwhelmingly FP noise; further gate lowering hurts holdout / FP.

## Promotion decision

**KEEP / confirm edge_threshold=0.40** under motion-relink OFF.  
Do **not** promote 0.35/0.30/0.25.  
Config: `configs/experiments/recipe_c_motion_off_edge_0_40_det0_96875.yaml`  
Historical motion-ON / edge-0.5 YAMLs unchanged.

## Bottleneck after edge 0.40 (vs motion-off edge 0.50)

| cause | fixed 0.50 | fixed 0.40 | hold 0.50 | hold 0.40 |
|-------|-----------:|-----------:|----------:|----------:|
| detection_miss | 37 | 37 | 35 | 35 |
| scorer_ranking | 75 | 70 | 25 | 26 |
| candidate_threshold | 48 | **13** | 16 | **3** |
| ilp_global | 21 | 9 | 12 | 9 |
| postprocessing_removed | 23 | 19 | 14 | 8 |
| postprocessing_rematch | 13 | 18 | 5 | 6 |

At edge 0.40: remaining `postprocessing_removed` are **100% missing-endpoint** (short-track cleanup).  
Ranking share ~54% fixed / 50% holdout → class B (not yet retrain-ready ≥60%).

## Next experiment (exactly one)

**Fresh instrumented candidate-edge capture on the promoted recipe**  
(motion OFF + edge 0.40) for fixed-8 + holdout, to validate whether ranking is now the actionable bottleneck before any scorer retrain.  
Do not reopen short-track OFF (already rejected) or further gate sweeps.

Artifacts: `outputs/experiments/edge_gate_sweep_motion_off_v1/`,  
`outputs/experiments/bottleneck_rediagnose_edge_0_40_v1/`
