#!/usr/bin/env python3
"""Offline rank-2 ordinary-association near-miss diagnosis from fresh captures."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

VOXEL_SCALE = np.array([1.625, 0.40625, 0.40625], dtype=np.float64)


def _load_nodes(cap_ds: Path) -> pd.DataFrame:
    with np.load(cap_ds / "nodes.npz") as data:
        return pd.DataFrame(
            {
                "node_id": data["node_id"].astype(np.int64),
                "t": data["t"].astype(np.int64),
                "z": data["z"].astype(float),
                "y": data["y"].astype(float),
                "x": data["x"].astype(float),
            }
        )


def _coords(nodes: pd.DataFrame) -> dict[int, np.ndarray]:
    out = {}
    for r in nodes.itertuples(index=False):
        out[int(r.node_id)] = np.array([r.z, r.y, r.x], dtype=np.float64)
    return out


def _dist_um(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm((a - b) * VOXEL_SCALE))


def _pairwise_rows(
    split: str,
    diag: pd.DataFrame,
    cap_root: Path,
) -> pd.DataFrame:
    """For each ranking failure (esp rank-2), join GT + best competitor features from npz."""
    rows: list[dict[str, Any]] = []
    fails = diag[diag["cause"] == "scorer_ranking"].copy()
    # also sample successful rank-1 for contrast
    succ = diag[(diag["cause"] == "correct") & (diag["target_rank"] == 1)].copy()
    # stratified sample successes for manageable size
    succ_sample = (
        succ.groupby("dataset", group_keys=False)
        .apply(lambda g: g.sample(n=min(len(g), 80), random_state=0))
        .reset_index(drop=True)
    )

    targets = pd.concat(
        [
            fails.assign(cohort="ranking_fail"),
            succ_sample.assign(cohort="success_r1"),
        ],
        ignore_index=True,
    )

    for dataset, g in targets.groupby("dataset"):
        ds_dir = cap_root / str(dataset)
        if not ds_dir.exists():
            continue
        nodes = _load_nodes(ds_dir)
        coords = _coords(nodes)
        # index captures by frame pair
        npz_by_t: dict[int, Path] = {}
        for f in ds_dir.glob("t*_to_t*.npz"):
            # t000_to_t001
            t_src = int(f.name.split("_")[0][1:])
            npz_by_t[t_src] = f

        for r in g.itertuples(index=False):
            t = int(r.t)
            fpath = npz_by_t.get(t)
            if fpath is None:
                continue
            gt_s = int(r.pred_source_id) if pd.notna(r.pred_source_id) else int(r.gt_source_id)
            gt_t = int(r.pred_target_id) if pd.notna(r.pred_target_id) else int(r.gt_target_id)
            with np.load(fpath) as data:
                sources = data["source_id"].astype(np.int64)
                targets_ = data["target_id"].astype(np.int64)
                # all candidates for this target
                mask_t = targets_ == gt_t
                if not mask_t.any():
                    continue
                idx = np.flatnonzero(mask_t)
                # find GT
                gt_mask = sources[idx] == gt_s
                if not gt_mask.any():
                    continue
                gt_i = idx[np.flatnonzero(gt_mask)[0]]
                # best competitor = highest blended_prob among others
                other = idx[sources[idx] != gt_s]
                if len(other) == 0:
                    continue
                best_o = other[int(np.argmax(data["blended_prob"][other]))]

                def feat(i: int, prefix: str) -> dict[str, Any]:
                    sid = int(sources[i])
                    tid = int(targets_[i])
                    d = {
                        f"{prefix}_source_id": sid,
                        f"{prefix}_target_id": tid,
                        f"{prefix}_blended_prob": float(data["blended_prob"][i]),
                        f"{prefix}_blended_logit": float(data["blended_logit"][i]),
                        f"{prefix}_seed1_logit": float(data["seed1_logit"][i]),
                        f"{prefix}_seed2_logit": float(data["seed2_logit"][i]),
                        f"{prefix}_seed1_prob": float(data["seed1_prob"][i]),
                        f"{prefix}_seed2_prob": float(data["seed2_prob"][i]),
                        f"{prefix}_source_rank": int(data["source_rank"][i]),
                        f"{prefix}_target_rank": int(data["target_rank"][i]),
                    }
                    if sid in coords and tid in coords:
                        d[f"{prefix}_dist_um"] = _dist_um(coords[sid], coords[tid])
                        delta = (coords[tid] - coords[sid]) * VOXEL_SCALE
                        d[f"{prefix}_dz_um"] = float(delta[0])
                        d[f"{prefix}_dy_um"] = float(delta[1])
                        d[f"{prefix}_dx_um"] = float(delta[2])
                    else:
                        d[f"{prefix}_dist_um"] = np.nan
                        d[f"{prefix}_dz_um"] = np.nan
                        d[f"{prefix}_dy_um"] = np.nan
                        d[f"{prefix}_dx_um"] = np.nan
                    return d

                row = {
                    "split": split,
                    "dataset": dataset,
                    "t": t,
                    "cohort": r.cohort,
                    "cause": r.cause,
                    "diag_target_rank": int(r.target_rank) if pd.notna(r.target_rank) else -1,
                    "diag_source_rank": int(r.source_rank) if pd.notna(r.source_rank) else -1,
                    "local_density_pred": float(r.local_density_pred)
                    if pd.notna(r.local_density_pred)
                    else np.nan,
                    "competitor_margin_diag": float(r.competitor_margin)
                    if pd.notna(r.competitor_margin)
                    else np.nan,
                    "n_cands_on_target": int(mask_t.sum()),
                }
                row.update(feat(gt_i, "gt"))
                row.update(feat(best_o, "comp"))
                # derived deltas: positive means GT better
                for key in [
                    "blended_prob",
                    "blended_logit",
                    "seed1_logit",
                    "seed2_logit",
                    "seed1_prob",
                    "seed2_prob",
                    "dist_um",
                ]:
                    row[f"d_{key}"] = row[f"gt_{key}"] - row[f"comp_{key}"]
                # seed disagreement on each side
                row["gt_seed_disagree"] = abs(row["gt_seed1_prob"] - row["gt_seed2_prob"])
                row["comp_seed_disagree"] = abs(row["comp_seed1_prob"] - row["comp_seed2_prob"])
                row["d_seed_disagree"] = row["gt_seed_disagree"] - row["comp_seed_disagree"]
                # which seed prefers GT?
                row["seed1_prefers_gt"] = int(row["d_seed1_logit"] > 0)
                row["seed2_prefers_gt"] = int(row["d_seed2_logit"] > 0)
                row["seeds_agree_gt"] = int(row["seed1_prefers_gt"] == 1 and row["seed2_prefers_gt"] == 1)
                row["seeds_agree_comp"] = int(row["seed1_prefers_gt"] == 0 and row["seed2_prefers_gt"] == 0)
                row["seed_conflict"] = int(row["seed1_prefers_gt"] != row["seed2_prefers_gt"])
                # distance advantage: GT closer?
                row["gt_closer"] = int(row["d_dist_um"] < 0) if pd.notna(row["d_dist_um"]) else -1
                # alpha sweep: would different blend flip ranking?
                # blended_logit = a*s1 + (1-a)*s2; compare gt vs comp
                alphas = np.linspace(0, 1, 21)
                flip_alphas = []
                for a in alphas:
                    gt_b = a * row["gt_seed1_logit"] + (1 - a) * row["gt_seed2_logit"]
                    cp_b = a * row["comp_seed1_logit"] + (1 - a) * row["comp_seed2_logit"]
                    if gt_b > cp_b:
                        flip_alphas.append(float(a))
                row["alpha_flip_any"] = int(len(flip_alphas) > 0)
                row["alpha_flip_count"] = len(flip_alphas)
                row["alpha_flip_min"] = min(flip_alphas) if flip_alphas else np.nan
                row["alpha_flip_max"] = max(flip_alphas) if flip_alphas else np.nan
                # at alpha=0.5 current
                row["gt_wins_at_0_5"] = int(row["d_blended_logit"] > 0)
                rows.append(row)
    return pd.DataFrame(rows)


def _summarize(pair: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {"n_rows": int(len(pair))}
    r2 = pair[(pair["cohort"] == "ranking_fail") & (pair["diag_target_rank"] == 2)]
    rf = pair[pair["cohort"] == "ranking_fail"]
    ok = pair[pair["cohort"] == "success_r1"]
    out["n_rank2"] = int(len(r2))
    out["n_ranking_fail"] = int(len(rf))
    out["n_success_sample"] = int(len(ok))

    def med(df, col):
        return float(np.nanmedian(df[col])) if len(df) and col in df else float("nan")

    def mean(df, col):
        return float(np.nanmean(df[col])) if len(df) and col in df else float("nan")

    for name, df in [("rank2", r2), ("rank_fail", rf), ("success", ok)]:
        out[name] = {
            "n": int(len(df)),
            "gt_prob_med": med(df, "gt_blended_prob"),
            "comp_prob_med": med(df, "comp_blended_prob"),
            "d_prob_med": med(df, "d_blended_prob"),
            "d_logit_med": med(df, "d_blended_logit"),
            "gt_dist_med": med(df, "gt_dist_um"),
            "comp_dist_med": med(df, "comp_dist_um"),
            "d_dist_med": med(df, "d_dist_um"),
            "gt_closer_rate": mean(df, "gt_closer"),
            "density_med": med(df, "local_density_pred"),
            "seed_conflict_rate": mean(df, "seed_conflict"),
            "seed1_prefers_gt_rate": mean(df, "seed1_prefers_gt"),
            "seed2_prefers_gt_rate": mean(df, "seed2_prefers_gt"),
            "alpha_flip_any_rate": mean(df, "alpha_flip_any"),
            "gt_seed_disagree_med": med(df, "gt_seed_disagree"),
            "comp_seed_disagree_med": med(df, "comp_seed_disagree"),
            "n_cands_med": med(df, "n_cands_on_target"),
            "d_seed1_logit_med": med(df, "d_seed1_logit"),
            "d_seed2_logit_med": med(df, "d_seed2_logit"),
        }
    # Distance-based rescue potential among rank2: GT closer but loses on score
    if len(r2):
        closer_lose = r2[(r2["gt_closer"] == 1) & (r2["d_blended_prob"] < 0)]
        farther_lose = r2[(r2["gt_closer"] == 0) & (r2["d_blended_prob"] < 0)]
        out["rank2_gt_closer_but_loses"] = {
            "n": int(len(closer_lose)),
            "rate": float(len(closer_lose) / len(r2)),
            "margin_med": float(np.nanmedian(-closer_lose["d_blended_prob"])) if len(closer_lose) else None,
            "d_dist_med": float(np.nanmedian(closer_lose["d_dist_um"])) if len(closer_lose) else None,
        }
        out["rank2_gt_farther_and_loses"] = {
            "n": int(len(farther_lose)),
            "rate": float(len(farther_lose) / len(r2)),
        }
        # How much distance bonus (in logit units) would flip?
        # heuristic: add -beta * dist_um to logit; need gt_logit - beta*gt_dist > comp_logit - beta*comp_dist
        # => beta * (comp_dist - gt_dist) > comp_logit - gt_logit
        # => beta * (-d_dist) > -d_logit  when d_dist = gt-comp
        betas = {}
        for beta in [0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0]:
            new_d = r2["d_blended_logit"] - beta * r2["d_dist_um"]
            # wait: score' = logit - beta*dist; d_score = d_logit - beta*d_dist
            flips = int((new_d > 0).sum())
            betas[str(beta)] = {"flips": flips, "rate": float(flips / len(r2))}
        out["rank2_distance_penalty_flip_rates"] = betas
        # density-conditioned distance flip at beta=0.2
        for dens_cut in [6, 8, 10]:
            sub = r2[r2["local_density_pred"] >= dens_cut]
            if len(sub) == 0:
                continue
            new_d = sub["d_blended_logit"] - 0.2 * sub["d_dist_um"]
            flips = int((new_d > 0).sum())
            out[f"rank2_dens>={dens_cut}_beta0.2_flips"] = {
                "n": int(len(sub)),
                "flips": flips,
                "rate": float(flips / len(sub)),
            }
        # seed alpha: flips if any alpha works
        out["rank2_alpha_rescue"] = {
            "any_alpha_flips": int(r2["alpha_flip_any"].sum()),
            "rate": float(r2["alpha_flip_any"].mean()),
        }
        # soft margin cases (<0.15) — ranking near misses truly near
        near = r2[(-r2["d_blended_prob"]) <= 0.15]
        out["rank2_soft_margin_le_0_15"] = {
            "n": int(len(near)),
            "rate": float(len(near) / len(r2)),
            "gt_closer_rate": float(near["gt_closer"].mean()) if len(near) else None,
            "alpha_flip_rate": float(near["alpha_flip_any"].mean()) if len(near) else None,
        }
    return out


def main() -> None:
    nfs = Path.home() / "biohub-outputs/experiments/candidate_capture_recipe_c_edge_0_40_v1"
    login_mirror = Path("/home/mc46451/biohub-cell-tracking/outputs/experiments/candidate_capture_recipe_c_edge_0_40_v1")
    out_dir = Path.home() / "biohub-outputs/analysis/rank2_near_miss_v1"
    out_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    summaries = {}
    for split in ["fixed8", "holdout8"]:
        # prefer NFS diagnostic; fall back to login mirror
        diag_path = nfs / split / "candidate_edge_diagnostic.csv"
        if not diag_path.exists():
            diag_path = login_mirror / split / "candidate_edge_diagnostic.csv"
        diag = pd.read_csv(diag_path)
        cap_root = nfs / split / "candidate_capture"
        pair = _pairwise_rows(split, diag, cap_root)
        pair.to_csv(out_dir / f"{split}_pairwise.csv", index=False)
        frames.append(pair)
        summaries[split] = _summarize(pair)
        print(split, "pairwise rows", len(pair), "rank2", summaries[split]["n_rank2"])

    all_pair = pd.concat(frames, ignore_index=True)
    all_pair.to_csv(out_dir / "combined_pairwise.csv", index=False)
    summaries["combined"] = _summarize(all_pair)
    (out_dir / "summary.json").write_text(json.dumps(summaries, indent=2))
    print(json.dumps(summaries, indent=2))
    print("WROTE", out_dir)


if __name__ == "__main__":
    main()
