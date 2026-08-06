"""Official-spec-lite utilities copied from the V106 local-CV cell.

SPDX-License-Identifier: Apache-2.0
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from math import isfinite

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

OFFICIAL_SPEC_NODE_PENALTY_A = 0.1
OFFICIAL_SPEC_DIVISION_WEIGHT = 0.1

@dataclass(frozen=True)
class SpecResult:
        edge_tp: int
        edge_fp: int
        edge_fn: int
        division_tp: int
        division_fp: int
        division_fn: int
        num_pred_nodes: int
        num_gt_nodes: int
        matched_gt_nodes: int


def _normalise_nodes(df: pd.DataFrame) -> pd.DataFrame:
        cols = ["node_id", "t", "z", "y", "x"]
        out = df.loc[:, cols].copy()
        out["node_id"] = out["node_id"].astype(np.int64)
        out["t"] = out["t"].astype(np.int64)
        for c in ("z", "y", "x"):
            out[c] = out[c].astype(np.float64)
        return out.drop_duplicates("node_id", keep="first").reset_index(drop=True)


def _normalise_edges(df: pd.DataFrame, nodes: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame(columns=["source_id", "target_id"])
        out = df.loc[:, ["source_id", "target_id"]].copy()
        out["source_id"] = out["source_id"].astype(np.int64)
        out["target_id"] = out["target_id"].astype(np.int64)
        out = out[out.source_id != out.target_id].drop_duplicates().reset_index(drop=True)
        valid = set(nodes.node_id.astype(int))
        out = out[out.source_id.isin(valid) & out.target_id.isin(valid)].copy()
        times = nodes.set_index("node_id")["t"].to_dict()
        out = out[
            out.apply(lambda r: int(times[int(r.target_id)]) == int(times[int(r.source_id)]) + 1, axis=1)
        ].copy()
        # Patched host scorer accepts at most two outgoing branches. Keep a deterministic
        # two-edge subset only for malformed diagnostic inputs; clean Notebook 124 is already <=2.
        out = out.sort_values(["source_id", "target_id"]).groupby("source_id", as_index=False).head(2)
        return out.reset_index(drop=True)


def _audit_prediction_nodes(nodes: pd.DataFrame) -> dict[str, int]:
        arr = nodes[["z", "y", "x"]].to_numpy(float)
        return {
            "negative_time_nodes": int((nodes.t < 0).sum()),
            "nonfinite_coordinate_nodes": int((~np.isfinite(arr).all(axis=1)).sum()),
            # Sentinel exploit coordinates are many orders outside the microscopy volume.
            "sentinel_coordinate_nodes": int((np.abs(arr) >= 1000.0).any(axis=1).sum()),
        }


def _node_match(
        pred_nodes: pd.DataFrame,
        gt_nodes: pd.DataFrame,
        scale=(1.625, 0.40625, 0.40625),
        max_distance_um: float = 7.0,
    ) -> tuple[dict[int, int], dict[int, int], dict[int, float]]:
        """Same-time optimal bipartite matching under a physical-distance gate."""
        p2g: dict[int, int] = {}
        g2p: dict[int, int] = {}
        distances: dict[int, float] = {}
        scale_arr = np.asarray(scale, dtype=np.float64)
        common_times = sorted(set(pred_nodes.t.astype(int)) & set(gt_nodes.t.astype(int)))
        for t in common_times:
            p = pred_nodes[pred_nodes.t == t]
            g = gt_nodes[gt_nodes.t == t]
            if p.empty or g.empty:
                continue
            pc = p[["z", "y", "x"]].to_numpy(float) * scale_arr
            gc = g[["z", "y", "x"]].to_numpy(float) * scale_arr
            dist = np.sqrt(((gc[:, None, :] - pc[None, :, :]) ** 2).sum(axis=2))
            big = max_distance_um * 1000.0 + 1.0
            cost = np.where(dist <= max_distance_um, dist, big)
            rr, cc = linear_sum_assignment(cost)
            gids = g.node_id.astype(int).to_numpy()
            pids = p.node_id.astype(int).to_numpy()
            for r, c in zip(rr, cc):
                d = float(dist[int(r), int(c)])
                if d > max_distance_um:
                    continue
                gid, pid = int(gids[int(r)]), int(pids[int(c)])
                p2g[pid] = gid
                g2p[gid] = pid
                distances[pid] = d
        return p2g, g2p, distances


def _adjacency(edges: pd.DataFrame):
        succ, pred = defaultdict(list), defaultdict(list)
        for r in edges.itertuples(index=False):
            s, t = int(r.source_id), int(r.target_id)
            succ[s].append(t); pred[t].append(s)
        return succ, pred


def _components(nodes: pd.DataFrame, edges: pd.DataFrame) -> dict[int, int]:
        adj = defaultdict(set)
        for r in edges.itertuples(index=False):
            s, t = int(r.source_id), int(r.target_id)
            adj[s].add(t); adj[t].add(s)
        comp, cid = {}, 0
        for n in nodes.node_id.astype(int):
            if n in comp:
                continue
            stack=[n]; comp[n]=cid
            while stack:
                x=stack.pop()
                for y in adj.get(x, ()):
                    if y not in comp:
                        comp[y]=cid; stack.append(y)
            cid += 1
        return comp


def _edge_counts(pred_edges, gt_edges, p2g):
        gt_set = {(int(r.source_id), int(r.target_id)) for r in gt_edges.itertuples(index=False)}
        gt_succ, gt_pred = _adjacency(gt_edges)
        tp_gt = set()
        fp = 0
        for r in pred_edges.itertuples(index=False):
            ps, pt = int(r.source_id), int(r.target_id)
            gs, gt = p2g.get(ps), p2g.get(pt)
            if gs is not None and gt is not None and (gs, gt) in gt_set and (gs, gt) not in tp_gt:
                tp_gt.add((gs, gt))
                continue
            evaluable = (gs is not None and len(gt_succ.get(gs, ())) > 0) or (
                gt is not None and len(gt_pred.get(gt, ())) > 0
            )
            if evaluable:
                fp += 1
        tp = len(tp_gt)
        fn = max(0, len(gt_set) - tp)
        return tp, fp, fn


def _kuhn_maximum_matching(valid_pairs: dict[int, set[int]]) -> dict[int, int]:
        """Map right(pred fork) -> left(GT division), maximum cardinality."""
        match_r: dict[int, int] = {}
        def dfs(left: int, seen: set[int]) -> bool:
            for right in sorted(valid_pairs.get(left, ())):
                if right in seen:
                    continue
                seen.add(right)
                if right not in match_r or dfs(match_r[right], seen):
                    match_r[right] = left
                    return True
            return False
        for left in sorted(valid_pairs):
            dfs(left, set())
        return match_r


def _division_counts(pred_nodes, pred_edges, gt_nodes, gt_edges, global_p2g, scale, max_distance_um):
        psucc, ppred = _adjacency(pred_edges)
        gsucc, gpred = _adjacency(gt_edges)
        gt_components = _components(gt_nodes, gt_edges)
        pred_forks = {n for n, ch in psucc.items() if len(ch) >= 2}
        gt_divisions = {n for n, ch in gsucc.items() if len(ch) == 2}
        pred_by_id = pred_nodes.set_index("node_id")
        gt_by_id = gt_nodes.set_index("node_id")

        valid_pairs: dict[int, set[int]] = defaultdict(set)
        local_candidate_forks: set[int] = set()

        for gd in sorted(gt_divisions):
            daughters = list(gsucc[gd])[:2]
            parent_side = {gd, *gpred.get(gd, [])}
            daughter_sets = []
            for d in daughters:
                daughter_sets.append({d, *gsucc.get(d, [])})
            local_gt_ids = sorted(parent_side | set().union(*daughter_sets))
            local_gt = gt_nodes[gt_nodes.node_id.isin(local_gt_ids)]
            if local_gt.empty:
                continue
            # Independent local matching, as specified for division windows.
            time_min, time_max = int(local_gt.t.min()), int(local_gt.t.max())
            local_pred = pred_nodes[(pred_nodes.t >= time_min) & (pred_nodes.t <= time_max)]
            lp2g, lg2p, _ = _node_match(local_pred, local_gt, scale, max_distance_um)
            parent_pred = {lg2p[g] for g in parent_side if g in lg2p}
            candidates = {p for p in parent_pred if p in pred_forks}
            for p in parent_pred:
                candidates.update(c for c in psucc.get(p, []) if c in pred_forks)
            local_candidate_forks.update(candidates)

            for fork in candidates:
                children = list(psucc.get(fork, []))
                if len(children) < 2:
                    continue
                branches = [{c, *psucc.get(c, [])} for c in children]
                # Parent anchor must be the fork or its immediate predecessor.
                parent_ok = fork in parent_pred or any(p in parent_pred for p in ppred.get(fork, []))
                if not parent_ok:
                    continue
                # Reject merged direct children and merged fallback grandchildren.
                if any(len(ppred.get(c, [])) != 1 for c in children):
                    continue
                evidence = [[False] * len(branches) for _ in range(2)]
                for di, dset in enumerate(daughter_sets):
                    for bi, bset in enumerate(branches):
                        evidence[di][bi] = any(lp2g.get(p) in dset for p in bset)
                # Two daughter lineages must map to two distinct predicted branches.
                branch_pair_ok = any(
                    evidence[0][b0] and evidence[1][b1]
                    for b0 in range(len(branches)) for b1 in range(len(branches)) if b0 != b1
                )
                if branch_pair_ok:
                    valid_pairs[gd].add(fork)

        matched = _kuhn_maximum_matching(valid_pairs)
        tp_forks = set(matched)
        tp = len(tp_forks)
        fn = max(0, len(gt_divisions) - tp)

        # Conservative FP evidence from the published patched specification.
        fp_forks: set[int] = set()
        for fork in pred_forks - tp_forks:
            g = global_p2g.get(fork)
            if g is not None and len(gsucc.get(g, [])) > 0:
                fp_forks.add(fork); continue
            if fork in local_candidate_forks:
                fp_forks.add(fork); continue
            children = list(psucc.get(fork, []))
            if any(len(ppred.get(c, [])) != 1 for c in children):
                fp_forks.add(fork); continue
            branch_components = []
            for c in children:
                direct_g = global_p2g.get(c)
                if direct_g is not None:
                    branch_components.append(gt_components.get(direct_g)); continue
                comps = {gt_components.get(global_p2g[gch]) for gch in psucc.get(c, []) if gch in global_p2g}
                comps.discard(None)
                if len(comps) == 1:
                    branch_components.append(next(iter(comps)))
            branch_components = [c for c in branch_components if c is not None]
            if len(set(branch_components)) >= 2:
                fp_forks.add(fork)
        return tp, len(fp_forks), fn


def official_spec_evaluate(pred_nodes, pred_edges, gt_nodes, gt_edges, scale=(1.625,0.40625,0.40625), max_distance_um=7.0):
        pred_nodes = _normalise_nodes(pred_nodes)
        gt_nodes = _normalise_nodes(gt_nodes)
        audit = _audit_prediction_nodes(pred_nodes)
        if any(audit.values()):
            raise ValueError(f"Prediction failed clean-node audit: {audit}")
        pred_edges = _normalise_edges(pred_edges, pred_nodes)
        gt_edges = _normalise_edges(gt_edges, gt_nodes)
        p2g, g2p, _ = _node_match(pred_nodes, gt_nodes, scale, max_distance_um)
        etp, efp, efn = _edge_counts(pred_edges, gt_edges, p2g)
        dtp, dfp, dfn = _division_counts(
            pred_nodes, pred_edges, gt_nodes, gt_edges, p2g, scale, max_distance_um
        )
        return SpecResult(etp, efp, efn, dtp, dfp, dfn, len(pred_nodes), len(gt_nodes), len(g2p))


def _jaccard(tp, fp, fn):
        den = int(tp) + int(fp) + int(fn)
        return float(tp) / den if den else 1.0


def official_spec_per_sample(r: SpecResult, estimated_total_nodes: float):
        edge_j = _jaccard(r.edge_tp, r.edge_fp, r.edge_fn)
        if np.isfinite(estimated_total_nodes) and estimated_total_nodes > 0:
            node_ratio = r.num_pred_nodes / float(estimated_total_nodes)
            factor = 1.0 - OFFICIAL_SPEC_NODE_PENALTY_A * (
                (r.num_pred_nodes - float(estimated_total_nodes)) / float(estimated_total_nodes)
            )
            adjusted = max(0.0, edge_j * factor)
        else:
            node_ratio, adjusted = float("nan"), edge_j
        return {
            "edge_jaccard": edge_j,
            "adj_edge_jaccard": adjusted,
            "division_jaccard": _jaccard(r.division_tp, r.division_fp, r.division_fn),
            "node_recall": (r.matched_gt_nodes / r.num_gt_nodes) if r.num_gt_nodes else 1.0,
            "total_node_ratio": node_ratio,
            "edge_weight": r.edge_tp + r.edge_fp + r.edge_fn,
        }


def official_spec_summarise(rows: list[dict]) -> dict:
        w = np.array([float(r.get("edge_weight", 0)) for r in rows], dtype=float)
        adj = np.array([float(r["adj_edge_jaccard"]) for r in rows], dtype=float)
        edge_adj = float(np.average(adj, weights=w)) if w.sum() > 0 else float(adj.mean())
        etp=sum(int(r["edge_tp"]) for r in rows); efp=sum(int(r["edge_fp"]) for r in rows); efn=sum(int(r["edge_fn"]) for r in rows)
        dtp=sum(int(r["division_tp"]) for r in rows); dfp=sum(int(r["division_fp"]) for r in rows); dfn=sum(int(r["division_fn"]) for r in rows)
        matched=sum(int(r["matched_gt_nodes"]) for r in rows); ngt=sum(int(r["num_gt_nodes"]) for r in rows)
        div_j=_jaccard(dtp,dfp,dfn)
        return {
            "score": edge_adj + OFFICIAL_SPEC_DIVISION_WEIGHT * div_j,
            "adj_edge_jaccard": edge_adj,
            "edge_jaccard_micro": _jaccard(etp,efp,efn),
            "division_jaccard": div_j,
            "edge_tp": etp, "edge_fp": efp, "edge_fn": efn,
            "division_tp": dtp, "division_fp": dfp, "division_fn": dfn,
            "node_recall": matched/ngt if ngt else 1.0,
        }


def _df_nodes(rows): return pd.DataFrame(rows, columns=["node_id","t","z","y","x"])


def _df_edges(rows): return pd.DataFrame(rows, columns=["source_id","target_id"])
