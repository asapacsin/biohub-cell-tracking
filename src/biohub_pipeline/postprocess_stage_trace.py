"""Stage-level tracing for filter_output_graph edge survival.

SPDX-License-Identifier: Apache-2.0
"""
from __future__ import annotations

from typing import Any

import numpy as np

from biohub_pipeline import postprocessing as pp


def _edge_set(edges: list[dict[str, object]]) -> set[tuple[int, int]]:
    return {(int(e["source_id"]), int(e["target_id"])) for e in edges}


def filter_output_graph_traced(
    nodes_by_id: dict[int, dict[str, object]],
    raw_edges: list[dict[str, object]],
    *,
    dataset: str | None = None,
    deepcenter_bundle: object | None = None,
    watch: set[tuple[int, int]] | None = None,
) -> tuple[dict[int, dict[str, object]], list[dict[str, object]], dict[str, int], dict[str, Any]]:
    """Run the production postprocess pipeline, recording when watched edges disappear."""
    watch = set(watch or ())
    present = {edge: True for edge in watch}
    first_loss: dict[tuple[int, int], str] = {}
    stage_snapshots: list[dict[str, Any]] = []

    def checkpoint(stage: str, nodes: dict[int, dict[str, object]], edges: list[dict[str, object]]) -> None:
        current = _edge_set(edges)
        node_ids = set(nodes)
        lost_now: list[tuple[int, int]] = []
        for edge in watch:
            if not present[edge]:
                continue
            if edge not in current:
                present[edge] = False
                first_loss[edge] = stage
                lost_now.append(edge)
        stage_snapshots.append(
            {
                "stage": stage,
                "n_edges": len(edges),
                "n_nodes": len(nodes),
                "watched_still_present": sum(1 for e in watch if present[e]),
                "watched_lost_here": len(lost_now),
                "lost_edges": lost_now,
                "lost_with_both_nodes_still_present": sum(
                    1 for s, t in lost_now if s in node_ids and t in node_ids
                ),
            }
        )

    # Keep keys identical to filter_output_graph so helper stages can mutate stats.
    stats: dict[str, int] = {
        "raw_edges": len(raw_edges),
        "dropped_nonconsecutive_edges": 0,
        "dropped_long_edges": 0,
        "dropped_multi_parent_edges": 0,
        "dropped_multi_child_edges": 0,
        "dropped_division_edges": 0,
        "gap_candidates": 0,
        "gap_pairs_selected": 0,
        "gap_reused_existing": 0,
        "gap_inserted_synthetic": 0,
        "gap_added_nodes": 0,
        "gap_added_edges": 0,
        "gap_skipped_node_cap": 0,
        "gap_density_nodes_scored": 0,
        "gap_density_candidates_expanded": 0,
        "gap_density_candidates_restricted": 0,
        "gap_density_selected_outside_base": 0,
        "gap_density_step_delta_milli_sum": 0,
        "gap_refined_synthetic": 0,
        "gap_refine_failed": 0,
        "gap_refine_rejected_shift": 0,
        "pruned_isolated_nodes": 0,
        "motion_relink_edges": 0,
        "motion_relink_tight_edges": 0,
        "motion_relink_relaxed_edges": 0,
        "motion_relink_frames": 0,
        "motion_relink_replaced_raw_edges": 0,
        "motion_relink_fallback_raw": 0,
        "motion_relink_skipped_large_frame": 0,
        "gap2_candidates": 0,
        "gap2_pairs_selected": 0,
        "gap2_added_nodes": 0,
        "gap2_added_edges": 0,
        "gap2_skipped_cap": 0,
        "safe_division_candidates": 0,
        "safe_divisions_added": 0,
        "safe_division_skipped_cap": 0,
        "deepcenter_gap_checked": 0,
        "deepcenter_gap_bypassed_strong_motion": 0,
        "deepcenter_gap_bypassed_synthetic_node": 0,
        "deepcenter_gap_accepted": 0,
        "deepcenter_gap_rejected": 0,
        "deepcenter_gap_missing": 0,
        "deepcenter_safe_div_checked": 0,
        "deepcenter_safe_div_accepted": 0,
        "deepcenter_safe_div_rejected": 0,
        "deepcenter_safe_div_missing": 0,
        "short_track_components_removed": 0,
        "short_track_nodes_removed": 0,
        "short_track_edges_removed": 0,
        "short_track_filter_skipped_all": 0,
        "short_track_rescue_triggered": 0,
        "short_track_rescue_components": 0,
        "short_track_rescue_nodes": 0,
        "short_track_rescue_budget": 0,
        "linefit_smoothed_nodes": 0,
        "linefit_skipped_nodes": 0,
    }

    edges: list[dict[str, object]] = []
    for edge in raw_edges:
        source = nodes_by_id.get(int(edge["source_id"]))
        target = nodes_by_id.get(int(edge["target_id"]))
        if source is None or target is None:
            continue
        if pp.OUTPUT_ENFORCE_NEXT_FRAME and int(target["t"]) != int(source["t"]) + 1:
            stats["dropped_nonconsecutive_edges"] += 1
            continue
        distance_um = pp.edge_distance_um(source, target)
        edge = dict(edge)
        edge["distance_um"] = distance_um
        if pp.OUTPUT_EDGE_MAX_UM > 0 and distance_um > pp.OUTPUT_EDGE_MAX_UM:
            stats["dropped_long_edges"] += 1
            continue
        edges.append(edge)
    checkpoint("distance_next_frame_filter", nodes_by_id, edges)

    if pp.OUTPUT_MOTION_RELINK:
        learned_edge_probs: dict[tuple[int, int], float] = {}
        for edge in edges:
            prob = edge.get("edge_prob")
            if prob is None:
                continue
            try:
                prob = float(prob)
            except (TypeError, ValueError):
                continue
            if np.isfinite(prob):
                key = (int(edge["source_id"]), int(edge["target_id"]))
                learned_edge_probs[key] = max(learned_edge_probs.get(key, float("-inf")), prob)
        motion_edges = pp.motion_relink_edges(nodes_by_id, stats, learned_edge_probs)
        if motion_edges:
            stats["motion_relink_replaced_raw_edges"] = len(edges)
            edges = motion_edges
        else:
            stats["motion_relink_fallback_raw"] = 1
    checkpoint("motion_relink", nodes_by_id, edges)

    if pp.OUTPUT_SINGLE_PARENT_REPAIR and edges:
        best_by_target: dict[int, dict[str, object]] = {}
        for edge in edges:
            target_id = int(edge["target_id"])
            prev = best_by_target.get(target_id)
            if prev is None or pp.edge_sort_key(edge) > pp.edge_sort_key(prev):
                best_by_target[target_id] = edge
        kept_ids = {id(edge) for edge in best_by_target.values()}
        stats["dropped_multi_parent_edges"] = sum(1 for edge in edges if id(edge) not in kept_ids)
        edges = [edge for edge in edges if id(edge) in kept_ids]
    checkpoint("single_parent_repair", nodes_by_id, edges)

    if pp.OUTPUT_SINGLE_CHILD_REPAIR and edges:
        best_by_source: dict[int, dict[str, object]] = {}
        for edge in edges:
            source_id = int(edge["source_id"])
            prev = best_by_source.get(source_id)
            if prev is None or pp.edge_sort_key(edge) > pp.edge_sort_key(prev):
                best_by_source[source_id] = edge
        kept_ids = {id(edge) for edge in best_by_source.values()}
        stats["dropped_multi_child_edges"] = sum(1 for edge in edges if id(edge) not in kept_ids)
        edges = [edge for edge in edges if id(edge) in kept_ids]
    checkpoint("single_child_repair", nodes_by_id, edges)

    repair_frame_cache: dict[int, np.ndarray] = {}
    deepcenter_heatmap_cache: dict[tuple[str, int], np.ndarray] = {}
    nodes_by_id, edges = pp.close_single_frame_gaps(
        nodes_by_id,
        edges,
        stats,
        dataset=dataset,
        deepcenter_bundle=deepcenter_bundle,
        frame_cache=repair_frame_cache,
        deepcenter_cache=deepcenter_heatmap_cache,
    )
    checkpoint("gap_close", nodes_by_id, edges)

    nodes_by_id, edges = pp.recover_strict_gap2(nodes_by_id, edges, stats, dataset=dataset)
    checkpoint("gap2_recovery", nodes_by_id, edges)

    edges = pp.add_safe_divisions_postlink(
        nodes_by_id,
        edges,
        stats,
        dataset=dataset,
        deepcenter_bundle=deepcenter_bundle,
        frame_cache=repair_frame_cache,
        deepcenter_cache=deepcenter_heatmap_cache,
    )
    checkpoint("safe_divisions", nodes_by_id, edges)

    if pp.OUTPUT_DIVISION_GEOMETRY_FILTER and edges:
        by_source: dict[int, list[dict[str, object]]] = {}
        for edge in edges:
            by_source.setdefault(int(edge["source_id"]), []).append(edge)

        filtered: list[dict[str, object]] = []
        for source_id, source_edges in by_source.items():
            if len(source_edges) <= 1:
                filtered.extend(source_edges)
                continue

            ranked = sorted(source_edges, key=pp.edge_sort_key, reverse=True)
            source = nodes_by_id[source_id]
            top1 = ranked[0]
            top2 = ranked[1]
            d1 = float(top1["distance_um"])
            d2 = float(top2["distance_um"])
            sister = pp.edge_distance_um(
                nodes_by_id[int(top1["target_id"])], nodes_by_id[int(top2["target_id"])]
            )
            valid_division = (
                max(d1, d2) <= pp.DIV_PARENT_MAX_UM
                and sister <= pp.DIV_SISTER_MAX_UM
                and int(nodes_by_id[int(top1["target_id"])]["t"]) == int(source["t"]) + 1
                and int(nodes_by_id[int(top2["target_id"])]["t"]) == int(source["t"]) + 1
            )
            if valid_division:
                filtered.extend([top1, top2])
                stats["dropped_division_edges"] += max(0, len(ranked) - 2)
            elif pp.DIV_DROP_TO_SINGLE_IF_BAD:
                filtered.append(top1)
                stats["dropped_division_edges"] += len(ranked) - 1
            else:
                filtered.extend(ranked)
        edges = filtered
    checkpoint("division_geometry_filter", nodes_by_id, edges)

    if pp.OUTPUT_PRUNE_ISOLATED:
        incident = {int(edge["source_id"]) for edge in edges} | {int(edge["target_id"]) for edge in edges}
        if incident:
            kept_nodes = {node_id: node for node_id, node in nodes_by_id.items() if node_id in incident}
            stats["pruned_isolated_nodes"] = len(nodes_by_id) - len(kept_nodes)
            nodes_by_id = kept_nodes
            edges = [
                edge
                for edge in edges
                if int(edge["source_id"]) in nodes_by_id and int(edge["target_id"]) in nodes_by_id
            ]
    checkpoint("prune_isolated", nodes_by_id, edges)

    nodes_by_id, edges = pp.filter_short_track_components(nodes_by_id, edges, stats)
    checkpoint("short_track_filter", nodes_by_id, edges)

    nodes_by_id = pp.linefit_smooth_output_graph(nodes_by_id, edges, stats)
    checkpoint("linefit_smooth", nodes_by_id, edges)

    for edge in watch:
        if present[edge] and edge not in first_loss:
            first_loss[edge] = "survived"

    trace = {
        "stage_snapshots": stage_snapshots,
        "first_loss_stage": {f"{s}->{t}": stage for (s, t), stage in first_loss.items()},
        "first_loss_by_edge": first_loss,
    }
    return nodes_by_id, edges, stats, trace
