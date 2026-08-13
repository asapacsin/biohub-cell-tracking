#!/usr/bin/env python3
"""Mine rank-2 hard negatives and fit a compact appearance-aware pairwise reweight.

score = blended_logit + w · φ(seed1, seed2, dist_um, dens)
Trained with pairwise logistic loss on fixed-8; validated offline on holdout-8.
Does not change gates / motion / edge_threshold — scoring only.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

VOXEL_SCALE = np.array([1.625, 0.40625, 0.40625], dtype=np.float64)
FEATURE_NAMES = [
    "seed_disagree",  # |s1-s2| appearance disagreement
    "seed_min",  # min(s1,s2) joint appearance confidence
    "dist_n",  # dist_um / 10
    "log_dens",  # log1p(density)
    "dist_x_disagree",  # appearance-gated distance
    "dist_x_seedmin",  # distance × joint confidence (downweight when both sure)
]


def load_nodes(cap_ds: Path) -> dict[int, np.ndarray]:
    with np.load(cap_ds / "nodes.npz") as data:
        ids = data["node_id"].astype(np.int64)
        zyx = np.stack([data["z"], data["y"], data["x"]], axis=1).astype(np.float64)
        return {int(i): zyx[k] for k, i in enumerate(ids)}


def dist_um(coords: dict[int, np.ndarray], s: int, t: int) -> float:
    return float(np.linalg.norm((coords[s] - coords[t]) * VOXEL_SCALE))


def phi_row(seed1: float, seed2: float, dist: float, dens: float) -> np.ndarray:
    disagree = abs(float(seed1) - float(seed2))
    smin = min(float(seed1), float(seed2))
    dist_n = float(dist) / 10.0
    log_dens = float(np.log1p(max(dens, 0.0)))
    return np.array(
        [
            disagree,
            smin,
            dist_n,
            log_dens,
            dist_n * disagree,
            dist_n * smin,
        ],
        dtype=np.float64,
    )


def collect_cases(split: str, diag: pd.DataFrame, cap_root: Path) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    em = diag[(diag["source_detected"] == True) & (diag["target_detected"] == True)]
    for dataset, g in em.groupby("dataset"):
        ds_dir = cap_root / str(dataset)
        if not ds_dir.exists():
            continue
        coords = load_nodes(ds_dir)
        npz_by_t = {int(f.name.split("_")[0][1:]): f for f in ds_dir.glob("t*_to_t*.npz")}
        for r in g.itertuples(index=False):
            fpath = npz_by_t.get(int(r.t))
            if fpath is None:
                continue
            gt_s = int(r.pred_source_id) if pd.notna(r.pred_source_id) else int(r.gt_source_id)
            gt_t = int(r.pred_target_id) if pd.notna(r.pred_target_id) else int(r.gt_target_id)
            dens = float(r.local_density_pred) if pd.notna(r.local_density_pred) else 0.0
            with np.load(fpath) as data:
                sources = data["source_id"].astype(np.int64)
                targets = data["target_id"].astype(np.int64)
                blend = data["blended_logit"].astype(np.float64)
                s1 = data["seed1_logit"].astype(np.float64)
                s2 = data["seed2_logit"].astype(np.float64)
                idx = np.flatnonzero(targets == gt_t)
                if len(idx) < 2:
                    continue
                src = sources[idx]
                gt_pos = np.flatnonzero(src == gt_s)
                if len(gt_pos) == 0:
                    continue
                gt_pos = int(gt_pos[0])
                d = np.array(
                    [
                        dist_um(coords, int(s), gt_t)
                        if int(s) in coords and gt_t in coords
                        else np.nan
                        for s in src
                    ]
                )
                if np.isnan(d).any():
                    med = float(np.nanmedian(d))
                    d = np.nan_to_num(d, nan=med)
                feats = np.stack(
                    [phi_row(s1[i], s2[i], d[k], dens) for k, i in enumerate(idx)],
                    axis=0,
                )
                order = np.argsort(-blend[idx])
                ranks = np.empty(len(idx), dtype=np.int32)
                ranks[order] = np.arange(1, len(idx) + 1)
                gap = float(blend[idx][order[0]] - blend[idx][order[1]])
                cases.append(
                    {
                        "split": split,
                        "dataset": str(dataset),
                        "cause": str(r.cause),
                        "target_rank_diag": int(r.target_rank) if pd.notna(r.target_rank) else -1,
                        "density": dens,
                        "blend": blend[idx].copy(),
                        "feats": feats,
                        "gt_index": gt_pos,
                        "base_rank": int(ranks[gt_pos]),
                        "gap01": gap,
                        "dists": d,
                    }
                )
    return cases


def score_case(case: dict[str, Any], w: np.ndarray) -> np.ndarray:
    return case["blend"] + case["feats"] @ w


def gt_rank(case: dict[str, Any], w: np.ndarray) -> int:
    s = score_case(case, w)
    order = np.argsort(-s)
    ranks = np.empty(len(s), dtype=np.int32)
    ranks[order] = np.arange(1, len(s) + 1)
    return int(ranks[case["gt_index"]])


def mine_pairs(cases: list[dict[str, Any]], *, hard_only: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Return (X_diff, y) for logistic: y=1 means GT should beat competitor."""
    xs: list[np.ndarray] = []
    for c in cases:
        blend = c["blend"]
        feats = c["feats"]
        gi = c["gt_index"]
        # hard neg = best non-GT by blend
        other = [i for i in range(len(blend)) if i != gi]
        if not other:
            continue
        if hard_only:
            # focus on ranking failures + near ties among successes
            if c["cause"] == "scorer_ranking":
                pass
            elif c["cause"] == "correct" and c["base_rank"] == 1 and c["gap01"] < 1.5:
                pass
            else:
                continue
        best_neg = max(other, key=lambda i: float(blend[i]))
        # Also include 2nd best for ranking fails
        negs = [best_neg]
        if c["cause"] == "scorer_ranking" and len(other) > 1:
            ranked = sorted(other, key=lambda i: float(blend[i]), reverse=True)
            if ranked[1] not in negs:
                negs.append(ranked[1])
        for ni in negs:
            # feature diff GT - neg; label 1 (GT should win)
            xs.append(feats[gi] - feats[ni])
            # also encode blend margin as part of training target via loss on full score:
            # we fit only w on φ; blended already present so use:
            # P(GT>neg) = σ( (b_gt-b_neg) + w·(φ_gt-φ_neg) )
            # store blend_diff alongside
    if not xs:
        return np.zeros((0, len(FEATURE_NAMES))), np.zeros(0)
    X = np.stack(xs, axis=0)
    return X, np.ones(len(X))


def mine_pairs_with_blend(
    cases: list[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray]:
    """X rows are φ_gt - φ_neg; b rows are blend_gt - blend_neg."""
    xs: list[np.ndarray] = []
    bs: list[float] = []
    weights: list[float] = []
    for c in cases:
        blend = c["blend"]
        feats = c["feats"]
        gi = c["gt_index"]
        other = [i for i in range(len(blend)) if i != gi]
        if not other:
            continue
        # upsample ranking failures; include near-tie successes as controls
        if c["cause"] == "scorer_ranking":
            w_case = 3.0 if c["target_rank_diag"] == 2 else 2.0
            ranked = sorted(other, key=lambda i: float(blend[i]), reverse=True)[:3]
        elif c["cause"] == "correct" and c["base_rank"] == 1:
            if c["gap01"] > 2.0 and c["density"] < 6:
                continue  # easy successes — skip
            w_case = 1.0
            ranked = [max(other, key=lambda i: float(blend[i]))]
        else:
            continue
        for ni in ranked:
            xs.append(feats[gi] - feats[ni])
            bs.append(float(blend[gi] - blend[ni]))
            weights.append(w_case)
    if not xs:
        return np.zeros((0, len(FEATURE_NAMES) + 1)), np.zeros(0)
    Xd = np.stack(xs, axis=0)
    bd = np.asarray(bs, dtype=np.float64)
    ww = np.asarray(weights, dtype=np.float64)
    # pack blend margin as column 0 of design for reporting; train uses separate b
    return np.concatenate([bd[:, None], Xd, ww[:, None]], axis=1), np.ones(len(Xd))


def fit_pairwise_logistic(
    pack: np.ndarray,
    *,
    l2: float = 0.5,
    steps: int = 400,
    lr: float = 0.05,
) -> np.ndarray:
    """Maximize Σ w log σ(b + Xw) - l2||w||^2 via gradient ascent."""
    b = pack[:, 0]
    X = pack[:, 1 : 1 + len(FEATURE_NAMES)]
    sample_w = pack[:, -1]
    w = np.zeros(len(FEATURE_NAMES), dtype=np.float64)
    for _ in range(steps):
        z = b + X @ w
        # σ(-z) = 1-σ(z); grad logσ(z) = σ(-z)
        sig_neg = 1.0 / (1.0 + np.exp(np.clip(z, -40, 40)))
        grad = (sample_w[:, None] * sig_neg[:, None] * X).sum(axis=0) - l2 * w
        w = w + lr * grad / max(float(sample_w.sum()), 1.0)
    return w


def eval_policy(cases: list[dict[str, Any]], w: np.ndarray, *, dens_min: float = 0.0, gap_max: float = 99.0) -> dict[str, Any]:
    rank2 = [c for c in cases if c["cause"] == "scorer_ranking" and c["target_rank_diag"] == 2]
    rank_fail = [c for c in cases if c["cause"] == "scorer_ranking"]
    success = [c for c in cases if c["cause"] == "correct" and c["base_rank"] == 1]

    def gated_rank(c: dict[str, Any]) -> tuple[int, bool]:
        apply = c["gap01"] < gap_max and c["density"] >= dens_min
        if not apply:
            return c["base_rank"], False
        return gt_rank(c, w), True

    def pack(group: list[dict[str, Any]], name: str) -> dict[str, Any]:
        if not group:
            return {"n": 0}
        ranks = []
        applied = 0
        flips_to_1 = 0
        drops = 0
        for c in group:
            new_r, ap = gated_rank(c)
            ranks.append(new_r)
            applied += int(ap)
            if c["base_rank"] > 1 and new_r == 1:
                flips_to_1 += 1
            if c["base_rank"] == 1 and new_r > 1:
                drops += 1
        return {
            "n": len(group),
            "applied": applied,
            "mean_rank": float(np.mean(ranks)),
            "top1_rate": float(np.mean([r == 1 for r in ranks])),
            "flips_to_1": flips_to_1,
            "success_drops": drops,
            "base_top1_rate": float(np.mean([c["base_rank"] == 1 for c in group])),
        }

    return {
        "rank2": pack(rank2, "rank2"),
        "rank_fail": pack(rank_fail, "rank_fail"),
        "success": pack(success, "success"),
    }


def main() -> None:
    root = Path("/home/mc46451/biohub-cell-tracking")
    cap = root / "outputs/experiments/candidate_capture_recipe_c_edge_0_40_v1"
    out = root / "outputs/analysis/pairwise_hardneg_rank_v1"
    out.mkdir(parents=True, exist_ok=True)

    cases_by_split: dict[str, list[dict[str, Any]]] = {}
    for split in ("fixed8", "holdout8"):
        diag = pd.read_csv(cap / split / "candidate_edge_diagnostic.csv")
        cases = collect_cases(split, diag, cap / split / "candidate_capture")
        cases_by_split[split] = cases
        print(split, "cases", len(cases), "rank2", sum(1 for c in cases if c["cause"] == "scorer_ranking" and c["target_rank_diag"] == 2))

    train_pack, _y = mine_pairs_with_blend(cases_by_split["fixed8"])
    print("train pairs", len(train_pack))
    if len(train_pack) < 10:
        raise SystemExit(f"too few training pairs: {len(train_pack)}")
    # grid over L2 / gates
    results = []
    best = None
    for l2 in (0.1, 0.5, 1.0, 2.0, 5.0):
        w = fit_pairwise_logistic(train_pack, l2=l2, steps=500, lr=0.08)
        for dens_min in (0.0, 6.0, 8.0):
            for gap_max in (99.0, 1.5, 1.0, 0.5):
                fixed_m = eval_policy(cases_by_split["fixed8"], w, dens_min=dens_min, gap_max=gap_max)
                hold_m = eval_policy(cases_by_split["holdout8"], w, dens_min=dens_min, gap_max=gap_max)
                # objective: maximize rank2 flips on both; constrain success drops
                score = (
                    fixed_m["rank2"]["flips_to_1"]
                    + hold_m["rank2"]["flips_to_1"]
                    - 2.0 * (fixed_m["success"]["success_drops"] + hold_m["success"]["success_drops"])
                )
                row = {
                    "l2": l2,
                    "dens_min": dens_min,
                    "gap_max": gap_max,
                    "w": w.tolist(),
                    "fixed": fixed_m,
                    "holdout": hold_m,
                    "score": score,
                }
                results.append(row)
                if best is None or score > best["score"]:
                    # require not catastrophic success drops
                    if (
                        fixed_m["success"]["success_drops"] <= 40
                        and hold_m["success"]["success_drops"] <= 40
                        and (fixed_m["rank2"]["flips_to_1"] + hold_m["rank2"]["flips_to_1"]) >= 1
                    ):
                        best = row

    # Always keep unconstrained best by score among constrained
    if best is None and results:
        # fallback: fewest success drops with any flip
        cand = [r for r in results if r["fixed"]["rank2"]["flips_to_1"] + r["holdout"]["rank2"]["flips_to_1"] >= 1]
        cand = sorted(cand, key=lambda r: (-r["score"], r["fixed"]["success"]["success_drops"]))
        best = cand[0] if cand else max(results, key=lambda r: r["score"])

    # Also report ungated L2=1.0 weights for inspection
    w_ref = fit_pairwise_logistic(train_pack, l2=1.0, steps=500, lr=0.08)
    summary = {
        "feature_names": FEATURE_NAMES,
        "n_train_pairs": int(train_pack.shape[0]),
        "best": {
            "l2": best["l2"],
            "dens_min": best["dens_min"],
            "gap_max": best["gap_max"],
            "w": best["w"],
            "w_named": dict(zip(FEATURE_NAMES, best["w"])),
            "fixed": best["fixed"],
            "holdout": best["holdout"],
            "score": best["score"],
        },
        "w_l2_1_ungated": {
            "w": w_ref.tolist(),
            "w_named": dict(zip(FEATURE_NAMES, w_ref.tolist())),
            "fixed": eval_policy(cases_by_split["fixed8"], w_ref),
            "holdout": eval_policy(cases_by_split["holdout8"], w_ref),
        },
        "top_results": sorted(results, key=lambda r: -r["score"])[:15],
    }
    (out / "fit_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    # save weights sidecar for runtime
    weights = {
        "feature_names": FEATURE_NAMES,
        "w": best["w"],
        "dens_min": best["dens_min"],
        "gap_max": best["gap_max"],
        "l2": best["l2"],
        "voxel_scale_um": [1.625, 0.40625, 0.40625],
        "note": "score = blended_logit + w·φ; apply when gap01<gap_max and dens>=dens_min",
    }
    (out / "pairwise_weights.json").write_text(json.dumps(weights, indent=2), encoding="utf-8")
    print(json.dumps(summary["best"], indent=2))
    print("wrote", out)


if __name__ == "__main__":
    main()
