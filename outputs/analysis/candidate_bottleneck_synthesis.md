# Candidate-edge bottleneck synthesis

Date: 2026-08-09

## Fixed-8 (generalization-safe recipe C)

- Score **0.884746427159** (bit-identical to control; instrumentation neutral)
- Classification: **D. HYPOTHESIS REJECTED**
- Causal errors: 193
- Mechanism shares: ranking **37.3%**, postprocessing_removed **29.5%**,
  candidate_threshold 16.1%, ilp_global 10.9%, rematch 6.2%
- Artifacts: `outputs/experiments/candidate_edge_bottleneck_v1/`

## Holdout-8 (same recipe; independent sequences)

- Score **0.959013051622** (bit-identical to prior holdout det=0.96875 run)
- Classification: **D. HYPOTHESIS REJECTED**
- Causal errors: 67
- Mechanism shares: postprocessing_removed **34.3%**, ranking **32.8%**,
  ilp_global 13.4%, candidate_threshold 11.9%, rematch 7.5%
- Artifacts: `outputs/experiments/candidate_edge_bottleneck_holdout8/`

## Decision

Do **not** retrain the edge scorer as the next production lever. Scorer ranking is
important but not dominant under the predeclared rule on either set. On holdout,
**postprocessing removal** is the largest single causal class.

**Next experiment (postprocess-only, no re-inference):** hold saved raw GEFFs fixed and
test one targeted retention change (most likely short-track filter / rescue settings),
evaluated on fixed-8 plus the same holdout-8. Keep the production submission recipe
unchanged until that ablation promotes.
