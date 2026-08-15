# Fixed-8 error taxonomy

Date: 2026-08-08  
Control: local recipe C two-seed α=0.5, det=0.960, safe-div ON, DeepCenter OFF  
Score: **0.887978610299** | edges 3885/273/218 | div 0/15/7

## Per-dataset error burden (control)

| dataset | score | edge FP/FN | div FP/FN | burden (FP+FN) |
|---------|------:|------------|-----------|---------------:|
| 6bba_05db0fb1 | 0.837193 | 133/81 | 6/3 | **223** |
| 6bba_fc83837d | 0.846920 | 76/80 | 2/0 | **158** |
| 44b6_e57ff5c6 | 0.720200 | 31/32 | 1/0 | **64** |
| 6bba_05b6850b | 0.970771 | 18/11 | 3/0 | 32 |
| 6bba_969618f6 | 0.963447 | 9/8 | 2/3 | 22 |
| 44b6_0b24845f | 0.936839 | 3/2 | 0/0 | 5 |
| 44b6_0113de3b | 0.905390 | 2/3 | 0/0 | 5 |
| 44b6_341df25f | 0.978581 | 1/1 | 1/1 | 4 |

Focus datasets from prior sensitivity (`44b6_e57ff5c6`, `44b6_341df25f`, `44b6_0b24845f`): only `e57ff5c6` remains a large absolute-error locus; the other two are near-ceiling and mainly matter for relative ablations (gap-2 / threshold / safe-div).

## Top recurring failure classes

### 1. Ordinary association errors on dense 6bba sequences (dominant)

- **Where:** `6bba_05db0fb1` + `6bba_fc83837d` account for **~78%** of edge FP+FN (381/491).
- **Signature:** large balanced FP and FN (identity switches / wrong parents), not pure detection sparsity.
- **Evidence:** node recall is already high in prior V106 controls; score gaps track edge Jaccard.
- **Recoverable?** Unknown without link-level audit; not solvable by threshold-only moves (0.960 already best in local sweep).
- **Risk of naive heuristics:** high FP if loosening link distance / gap recovery (gap-2 already rejected).

### 2. Safe-division FP with zero division TP

- **Where:** aggregate div TP/FP/FN = **0/15/7** under control.
- **Signature:** all predicted divisions from safe-div are false; 7 true divisions remain missed.
- **Ablations:**
  - early safe-div OFF → 0/0/7, score **0.888330** (+0.000351), 6/8 datasets up, but short-track cascade drops ~221 ordinary edges
  - late safe-div strip → 0/0/7, score **0.888208** (+0.000230), ordinary edges bit-identical to control
- **Interpretable win:** removing FP divisions; no TP recovered.
- **Generalization gate:** Phase 4 holdout must confirm FP-heavy / zero-TP outside fixed-8 before promoting OFF.

### 3. Detection/threshold-sensitive association on sparse 44b6 sequences

- **Where:** `44b6_e57ff5c6` (burden 64) and threshold deltas on `44b6_341df25f`.
- **det=0.960 vs 0.96875 (fixed-8):** aggregate +0.00323 for 0.960, but **4/8 win each**; gain concentrated in `e57ff5c6` (+0.0174) and `341df25f` (+0.0133).
- **Classification tendency on fixed-8 alone:** **WEAK–MODERATE** (not broad). Holdout required.

## Ablation-linked failure notes

| Intervention | Effect | Failure addressed | Decision |
|--------------|--------|-------------------|----------|
| gap-2 ON | +0.000162; 3/8 up; large loss on `44b6_0b24845f` (−0.0186) | some FN edges | **REJECT** |
| DeepCenter gating | score equals early safe-div OFF; rejects all safe-div | div FP via veto | **DISCARD** (no independent value) |
| Public dual-seed recipe A | fixed-8 **0.877745** (−0.01023 vs control) | n/a — different ensemble | not competitive locally |
| Public vs control worst datasets | `e57ff5c6` −0.084; `0b24845f` −0.047 | public blend/postprocess mismatch | investigate isolation |

Public recipe postprocess stats (this run): `safe_divisions_added=412`, `short_track_nodes_removed=6487`, DeepCenter gap checks unused (0 accept/reject), gap2=0.

## Proposed next interventions (do not all run)

| # | Intervention | Failure addressed | Potentially recoverable | FP risk | Inference? | Info value |
|---|--------------|-------------------|-------------------------|---------|------------|------------|
| A | **Public GEFFs + local recipe-C postprocess** | isolate whether public loss is blend vs postprocess | clarifies −0.01 gap | low | postprocess only | **High** — one conceptual variable |
| B | Promote late safe-div strip (or early OFF) if holdout agrees | div FP 15→0 | +0.0002–0.00035 | low on div; cascade risk if early OFF | postprocess | High for submission choice |
| C | Link-level audit on `6bba_05db0fb1` wrong-parent edges | association class #1 | unknown | n/a | analysis | High before any new heuristic |
| D | New gap/distance heuristic | residual FN | unknown | high (gap-2 precedent) | postprocess | Low now — rejected family |
| E | Dense retune of det threshold | class #3 | already swept | overfitting | inference | **Do not run** |

**Selected Phase-6 candidate:** A (public GEFFs × local postprocess), after holdout Phases 3–4 complete. Promote only if it isolates the gap and/or yields δ≥+0.001 vs public with clean control.
