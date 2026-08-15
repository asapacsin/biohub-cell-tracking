# Autonomous experiment cycle report (2026-08-09 → 2026-08-10)

## Best scores obtained

| set | original recipe C (motion ON, edge 0.5) | **best now** | cumulative Δ |
|-----|----------------------------------------:|-------------:|-------------:|
| fixed-8 | 0.884746 | **0.918144** | **+0.033398** |
| holdout-8 | 0.959013 | **0.964673** | **+0.005660** |

Best recipe: α=0.5, det=0.96875, safe-div ON, short-track ON (min_len=6),
**motion_relink OFF**, **edge_threshold=0.40**.

## Experiments performed

1. Short-track OFF / min_len 4/3/2 — all worse → KEEP short-track.
2. Stage-attribute `postprocessing_removed` — motion_relink 80.7% / 56.5%.
3. Motion-relink OFF — fixed +0.0220, holdout +0.0013 → **PROMOTE**.
4. Single-parent OFF — score-noop.
5. Rediagnose under motion OFF — remaining pp_removed = 100% short-track; threshold next.
6. Edge threshold 0.45 — fixed +0.0085, holdout +0.0024 → **PROMOTE**.
7. Edge threshold 0.40 — fixed +0.0030, holdout +0.0019 → **PROMOTE**.

## Hypotheses accepted / rejected

| hypothesis | verdict |
|------------|---------|
| Disable/relax short-track to recover true tracks | **REJECTED** |
| Motion-relink causes harmful postprocess removals | **ACCEPTED** |
| Single-parent repair is a harmful removal stage | **REJECTED** (noop) |
| Remaining pp_removed after motion OFF is short-track | **ACCEPTED** |
| Lower edge_threshold 0.5→0.45 net-positive | **ACCEPTED** |
| Lower edge_threshold 0.45→0.40 net-positive | **ACCEPTED** |
| Retrain edge scorer as immediate next lever | **still rejected** (ranking not dominant) |

## Current bottleneck

Under the promoted recipe, pre-ILP association quality remains mixed:
ranking ~35–42% of causal ordinary-association errors historically, with
candidate_threshold partially addressed by 0.40. Do **not** retrain yet without
a fresh capture+taxonomy on the promoted recipe. Postprocess retention is no longer
the primary local lever (short-track remaining removals are net helpful).

## Production / recommended recipe

**Do not silently overwrite historical baseline YAMLs** (motion ON / edge 0.5).

Recommended local generalization-safe recipe:
`configs/experiments/recipe_c_motion_off_edge_0_40_det0_96875.yaml`

- ensemble α=0.5 (two-seed raw logits)
- detection_threshold=0.96875
- edge_threshold=0.40
- output_safe_divisions=true
- output_motion_relink=**false**
- output_filter_short_tracks=true, output_min_track_len=6
- gap2 OFF, DeepCenter OFF

Historical comparison baseline remains
`configs/sweeps/two_seed_det_thresh_0_96875.yaml` (motion ON, implicit edge 0.5).

## Recommended next major direction

1. Re-run instrumented candidate-edge capture on the **promoted** recipe
   (motion OFF + edge 0.40) for fixed-8 and holdout.
2. If ranking dominates (≥60% with margin), then hard-negative / ranking retrain.
3. Else prefer one calibrated ILP objective change or a single ranking-aware
   gate tweak — not another broad threshold sweep (returns already diminishing).

## Important artifact paths

- Experiment log: `outputs/analysis/experiment_log.md`
- Short-track: `outputs/experiments/shorttrack_ablation_det0_96875/`
- Stage attr: `outputs/experiments/postprocess_stage_attribution_v1/`
- Motion ablation: `outputs/experiments/ppstage_ablation_det0_96875/`
- Rediagnose: `outputs/experiments/bottleneck_rediagnose_motion_off_v1/`
- Edge 0.45: `outputs/experiments/edge_thresh_0_45_motion_off_v1/`
- Edge 0.40: `outputs/experiments/edge_thresh_0_40_motion_off_v1/`
- NFS raw GEFFs: `~/biohub-outputs/experiments/edge_thresh_0_40_motion_off_v1/{fixed8,holdout8}/raw_geff`
