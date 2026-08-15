# Final experiment report — Biohub cell tracking

Date: 2026-08-08  
Objective: strongest **defensible** Kaggle configuration without overfitting fixed-8.

## 1. Exact public ~0.911 recipe

Source: Pilkwang notebook `biohub-cell-tracking-two-seeds-logit-blend`, tag  
`selected_101_dual_seed_near_balanced_center_confirmed_synthetic_gap`.  
Full audit: `outputs/audit/public_two_seed_recipe.md`.

| Setting | Value | Evidence |
|---------|-------|----------|
| Seed1 / Seed2 SHA-256 | `12f6881e…` / `9bac2fa0…` | verified |
| Detection threshold | **0.96875** | verified (not 0.960) |
| Det blend | aligned logits, secondary weight **0.475** | verified |
| Edge blend | logits, weight **0.15**, mode **`low_margin_consensus`**, τ=0.48, temp=1.0, low-margin max=0.35 | verified |
| ILP disappearance | **1.5** | verified |
| TTA | spatial D4 | verified |
| safe-div | ON | verified |
| gap2 | OFF | verified |
| DeepCenter | gap gate ON (0.25 / 8.5 µm); safe-div veto OFF | verified |
| short-track rescue | OFF; min_track_len 6 | verified |
| Literal public LB 0.911 | — | **unknown** (not in artifacts) |

Local recipe C (`clean_v106_two_seed.yaml`, α=0.5 equal raw-logit blend) is **not** this recipe.

## 2. Local reproduction of public recipe

Output: `outputs/fixed8/public_two_seed_exact/`

| Metric | Public recipe A | Local det=0.960 control |
|--------|----------------:|------------------------:|
| Score | **0.877744529257** | 0.887978610299 |
| Edge TP/FP/FN | 3863/298/240 | 3885/273/218 |
| Div TP/FP/FN | 0/10/7 | 0/15/7 |
| Δ | **−0.010234** | — |

Worst losses vs control: `44b6_e57ff5c6` (−0.084), `44b6_0b24845f` (−0.047).

**Verdict:** Public recipe reproduced locally and is **materially worse** on fixed-8 than local recipe C. Local 0.960 is not an accidental match to the public pipeline.

## 3–4. Published threshold (0.96875) vs 0.960

Same local recipe C (α=0.5), identical postprocess; only `detection_threshold` changes.

### Fixed-8

| thresh | score | edge TP/FP/FN | datasets won |
|-------:|------:|---------------|--------------|
| 0.960 | **0.887979** | 3885/273/218 | 4/8 |
| 0.96875 | 0.884746 | 3878/283/225 | 4/8 |

Gain for 0.960 concentrated in `44b6_e57ff5c6` (+0.017) and `44b6_341df25f` (+0.013).

### Holdout-8 (unused for selecting 0.960)

Datasets: `44b6_0c582fdc, 44b6_0db75fae, 44b6_12dfb391, 44b6_144b256d, 6bba_062c8d37, 6bba_07477033, 6bba_07e24132, 6bba_085bf656`  
Artifacts: `outputs/holdout8/`

| thresh | score | edge TP/FP/FN | div TP/FP/FN | datasets won |
|-------:|------:|---------------|--------------|--------------|
| 0.960 | 0.958371 | 4023/94/103 | 0/9/5 | **1** |
| 0.96875 | **0.959013** | 4022/93/104 | 0/10/5 | **7** |

Δ(0.960 − 0.96875) = **−0.000642**

**Classification: REJECT** for promoting 0.960 — independent holdout favors the published threshold.

## 5. Safe-div ON vs OFF

### Fixed-8 (det=0.960)

| config | score | edge | div | notes |
|--------|------:|------|-----|-------|
| safe-div ON (control) | 0.887979 | 3885/273/218 | 0/15/7 | — |
| early OFF | **0.888330** | 3877/263/226 | 0/0/7 | 6/8 up; short-track cascade |
| late strip | 0.888208 | 3878/264/225 | 0/0/7 | ordinary edges bit-identical |

### Holdout-8 (det=0.960 GEFFs)

| config | score | edge | div |
|--------|------:|------|-----|
| safe ON | 0.958371 | 4023/94/103 | 0/9/5 |
| safe OFF | 0.958650 | 4018/88/108 | 0/0/5 |

Δ = **+0.000278**, but regressions: `44b6_0c582fdc` **−0.0136**, `44b6_0db75fae` **−0.0061** (also −0.0036 on `12dfb391`). Zero division TP persists; FP removed (9→0).

**Decision: do not promote safe-div OFF** for a generalization-safe submission. Fixed-8 gain is real but small; holdout shows severe per-dataset regressions. Prefer keeping safe-div ON, or late-strip only as an aggressive optional second submission.

## 6. Rejected experiments

| Experiment | Score / Δ | Why rejected |
|------------|-----------|--------------|
| gap-2 | 0.888141 (+0.000162) | 3/8 up; large loss on `44b6_0b24845f` (−0.019) |
| DeepCenter | 0.888330 (+0.000351) | Effect entirely from rejecting all safe-div; no independent value |
| late safe-div strip | 0.888208 (+0.000230) | Cleaner than early OFF but still fixed-8-tuned; not selected as primary |
| det=0.960 vs published | holdout −0.00064, 1/8 wins | **REJECT** promotion of 0.960 |
| Public recipe A on fixed-8 | 0.877745 (−0.010) | Inferior locally; LB 0.911 unverified |
| Phase-6 public GEFFs × local PP | 0.878148 (+0.00040 vs public) | Gap is ensemble/inference, not postprocess; δ≪0.001 vs control |

## 7. Best fixed-8 configuration

**Local recipe C, α=0.5, det=0.960, safe-div OFF (early)**  
Score **0.888330075629** | edges 3877/263/226 | div 0/0/7  

Runner-up (cleaner): late strip **0.888208**.

These maximize fixed-8 but fail holdout generalization gates for threshold and (for OFF) show large holdout dataset regressions.

## 8. Best GENERALIZATION-SAFE configuration

**Local recipe C (equal α=0.5 raw-logit blend)**  
- `detection_threshold`: **0.96875** (holdout-backed; matches published threshold)  
- `ensemble_alpha`: **0.5**  
- safe-div: **ON**  
- gap2: **OFF**  
- DeepCenter: **OFF**  
- ILP disappearance: **1.575** (local V106 default)  
- short-track rescue: local defaults (adaptive ON as in recipe C yaml)

Fixed-8 score under this config: **0.884746427159**  
Holdout-8 score: **0.959013051622**

## 9. Recommended Kaggle submission

Use §8 (generalization-safe):

```
configs: start from configs/sweeps/two_seed_det_thresh_0_96875.yaml
  (or equivalent: clean_v106_two_seed + detection_threshold=0.96875)
inference:
  ensemble_alpha: 0.5
  detection_threshold: 0.96875
  spatial_d4_tta: true
postprocessing:
  output_safe_divisions: true
  output_gap2_recovery: false
  use_deepcenter_veto: false
```

Rationale: holdout favors 0.96875; public threshold agreement; avoids safe-div OFF regressions; discards rejected heuristics.

## 10. Optional second submission

**A (diversity / fixed-8 aggressive):** recipe C, det=**0.960**, **late safe-div strip** (0.888208) — ordinary edges unchanged vs control; removes div FP without cascade.  
**B (LB chase, weak local evidence):** public dual-seed recipe A — only if chasing the unverified ~0.911 notebook path; fixed-8 evidence is strongly negative.

## 11. Remaining uncertainty

1. Literal public LB **0.911** never verified in artifacts.  
2. Public notebook postprocess may still differ in unmapped details (local `used_sources` safe-div fix; scripting environment). Phase 6 shows postprocess swap recovers only +0.0004 → ensemble dominates.  
3. Holdout-8 is one unused slice (191 available); not a full nested CV.  
4. Dominant residual errors are association mistakes on dense `6bba_05db0fb1` / `6bba_fc83837d` (~78% of fixed-8 edge FP+FN) — no safe heuristic identified.  
5. Division TP remains 0 everywhere tested; safe-div only adds FP.

## 12. Next experiment (only if clearly supported)

**Link-level audit** of wrong-parent edges on `6bba_05db0fb1` and `6bba_fc83837d` (analysis only; no new heuristic yet).  
Do **not** re-tune thresholds, resurrect gap-2/DeepCenter, or dense-search fixed-8.

No further automatic heuristic search in this workflow.
