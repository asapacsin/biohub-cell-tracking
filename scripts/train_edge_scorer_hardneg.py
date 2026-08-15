#!/usr/bin/env python3
"""Fine-tune two-seed edge transformers with pairwise hard-neg ranking.

UNet + detect_head stay frozen so detections match the production recipe.
Only SimpleNodeTransformer is updated. Features are indexed at capture
predicted-node coordinates (fixed-8 only). Holdout-8 is never used.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import zarr

from biohub_pipeline.association_density import read_geff_tables
from biohub_pipeline.edge_hardneg_mine import (
    FIXED8_DATASETS,
    mine_pairs_from_capture,
    summarize_pairs,
)
from biohub_pipeline.evaluation import _node_match

DOWNSAMPLE = (1, 4, 4)
MARGIN = 0.5
RANK_LOSS_WEIGHT = 1.0
BCE_LOSS_WEIGHT = 1.0
ONLINE_HARDNEG_WEIGHT = 0.5


def _add_support_paths(support_dir: Path) -> None:
    repo = support_dir / "repo"
    scripts = repo / "scripts"
    src = repo / "src"
    for path in (str(scripts), str(src)):
        if path not in sys.path:
            sys.path.insert(0, path)


def _load_frame(zarr_arr, t: int, target_shape: list[int], downsample: tuple[int, ...]):
    dz, dy, dx = downsample
    raw = zarr_arr[t, ::dz, ::dy, ::dx].astype(np.float32)
    frame = torch.from_numpy(raw)
    if list(frame.shape) != target_shape:
        frame = F.interpolate(
            frame[None, None], size=target_shape, mode="trilinear", align_corners=False
        )[0, 0]
    return frame


@dataclass
class CachedWindow:
    dataset: str
    t: int
    src_ids: np.ndarray
    tgt_ids: np.ndarray
    feat_src: torch.Tensor
    feat_tgt: torch.Tensor
    coords_src: torch.Tensor
    coords_tgt: torch.Tensor
    pos_src: torch.Tensor
    pos_tgt: torch.Tensor
    gt_matrix: torch.Tensor
    pair_index: list[tuple[int, int, int, float]]  # src_gt, src_neg, tgt, weight


def _pos_features(coords_tzyx: np.ndarray, image_shape: tuple[int, ...], extract_pos):
    return torch.from_numpy(extract_pos(coords_tzyx, image_shape))


def _index_map(ids: np.ndarray) -> dict[int, int]:
    return {int(n): i for i, n in enumerate(ids.tolist())}


@torch.no_grad()
def cache_dataset_windows(
    *,
    model,
    dataset: str,
    data_dir: Path,
    capture_ds: Path,
    pairs: pd.DataFrame,
    device: torch.device,
    extract_pos,
    max_match_um: float = 7.0,
    scale: tuple[float, float, float] = (1.625, 0.40625, 0.40625),
) -> list[CachedWindow]:
    from biohub_tracking.io import open_dataset

    with np.load(capture_ds / "nodes.npz") as data:
        nodes = pd.DataFrame(
            {
                "node_id": data["node_id"].astype(np.int64),
                "t": data["t"].astype(np.int64),
                "z": data["z"].astype(np.float64),
                "y": data["y"].astype(np.float64),
                "x": data["x"].astype(np.float64),
            }
        )
    gt = read_geff_tables(data_dir / f"{dataset}.geff")
    gt_nodes = gt.nodes.loc[:, ["node_id", "t", "z", "y", "x"]]
    gt_edges = gt.edges.loc[:, ["source_id", "target_id"]].drop_duplicates()
    _, g2p, _ = _node_match(nodes, gt_nodes, scale=scale, max_distance_um=max_match_um)

    ds = open_dataset(
        data_dir / dataset, normalize=False, load_image=False, downsample=DOWNSAMPLE
    )
    zarr_arr = zarr.open_group(str(ds.zarr_path), mode="r")["0"]
    q_low = float(ds.quantiles["0.001"])
    q_high = float(ds.quantiles["0.999"])
    T = int(ds.image_shape[0])
    image_shape = (2, *tuple(ds.image_shape[1:]))
    target_shape = list(ds.image_shape[1:])
    ds_pairs = pairs[pairs["dataset"] == dataset]
    pair_by_t: dict[int, pd.DataFrame] = {
        int(t): part for t, part in ds_pairs.groupby("t", sort=False)
    }

    cached: list[CachedWindow] = []
    model.eval()
    for t in range(T - 1):
        src_nodes = nodes[nodes["t"] == t]
        tgt_nodes = nodes[nodes["t"] == t + 1]
        if src_nodes.empty or tgt_nodes.empty:
            continue
        imgs = torch.stack(
            [
                _load_frame(zarr_arr, t, target_shape, DOWNSAMPLE),
                _load_frame(zarr_arr, t + 1, target_shape, DOWNSAMPLE),
            ]
        )
        imgs = ((imgs - q_low) / (q_high - q_low + 1e-6)).clamp(0.0)
        imgs = imgs.unsqueeze(0).to(device)
        unet_out, _ = model.encode(imgs)

        src_zyx = src_nodes[["z", "y", "x"]].to_numpy(np.float32)
        tgt_zyx = tgt_nodes[["z", "y", "x"]].to_numpy(np.float32)
        src_ds = src_zyx / np.array(DOWNSAMPLE, dtype=np.float32)
        tgt_ds = tgt_zyx / np.array(DOWNSAMPLE, dtype=np.float32)
        src_ids = src_nodes["node_id"].to_numpy(np.int64)
        tgt_ids = tgt_nodes["node_id"].to_numpy(np.int64)
        n_src, n_tgt = len(src_ids), len(tgt_ids)

        p_src = torch.from_numpy(src_ds).unsqueeze(0).to(device)
        p_tgt = torch.from_numpy(tgt_ds).unsqueeze(0).to(device)
        mask_s = torch.ones(1, n_src, dtype=torch.bool, device=device)
        mask_t = torch.ones(1, n_tgt, dtype=torch.bool, device=device)
        feat_src = model._index_features(unet_out[:, 0], p_src, mask_s).squeeze(0).cpu()
        feat_tgt = model._index_features(unet_out[:, 1], p_tgt, mask_t).squeeze(0).cpu()

        src_rel = np.column_stack([np.zeros(n_src, dtype=np.float32), src_ds])
        tgt_rel = np.column_stack([np.ones(n_tgt, dtype=np.float32), tgt_ds])
        pos_src = _pos_features(src_rel, image_shape, extract_pos)
        pos_tgt = _pos_features(tgt_rel, image_shape, extract_pos)
        coords_src = torch.from_numpy(src_zyx)  # original voxels, matching predict_edges
        coords_tgt = torch.from_numpy(tgt_zyx)

        src_index = _index_map(src_ids)
        tgt_index = _index_map(tgt_ids)
        gt_matrix = torch.zeros(n_src, n_tgt, dtype=torch.float32)
        for edge in gt_edges.itertuples(index=False):
            ps = g2p.get(int(edge.source_id))
            pt = g2p.get(int(edge.target_id))
            if ps is None or pt is None:
                continue
            si = src_index.get(int(ps))
            ti = tgt_index.get(int(pt))
            if si is not None and ti is not None:
                gt_matrix[si, ti] = 1.0

        pair_index: list[tuple[int, int, int, float]] = []
        if t in pair_by_t:
            for rec in pair_by_t[t].itertuples(index=False):
                si = src_index.get(int(rec.pred_gt_source_id))
                ni = src_index.get(int(rec.pred_neg_source_id))
                ti = tgt_index.get(int(rec.pred_target_id))
                if si is None or ni is None or ti is None:
                    continue
                pair_index.append((si, ni, ti, float(rec.weight)))

        cached.append(
            CachedWindow(
                dataset=dataset,
                t=t,
                src_ids=src_ids,
                tgt_ids=tgt_ids,
                feat_src=feat_src,
                feat_tgt=feat_tgt,
                coords_src=coords_src,
                coords_tgt=coords_tgt,
                pos_src=pos_src,
                pos_tgt=pos_tgt,
                gt_matrix=gt_matrix,
                pair_index=pair_index,
            )
        )
        del imgs, unet_out
        torch.cuda.empty_cache()
    return cached


def _bce_ranking_losses(
    logits: torch.Tensor,
    window: CachedWindow,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    """Return (rank_loss, bce_loss, n_rank_pairs, n_rank_correct)."""
    target = window.gt_matrix.to(device)
    active_rows = target.sum(dim=1) > 0
    active_cols = target.sum(dim=0) > 0
    mask = active_rows.unsqueeze(1) | active_cols.unsqueeze(0)
    if mask.any():
        probs = torch.softmax(logits, dim=0)
        bce = F.binary_cross_entropy(probs, target, reduction="none")
        p_t = probs * target + (1 - probs) * (1 - target)
        focal = ((1 - p_t) ** 2) * bce
        bce_loss = focal[mask].mean()
    else:
        bce_loss = logits.sum() * 0.0

    rank_terms: list[torch.Tensor] = []
    n_ok = 0
    seen: set[tuple[int, int, int]] = set()
    for si, ni, ti, weight in window.pair_index:
        key = (si, ni, ti)
        seen.add(key)
        diff = logits[si, ti] - logits[ni, ti]
        rank_terms.append(float(weight) * F.softplus(MARGIN - diff))
        n_ok += int((diff.detach() > 0).item())
    # Online hard-neg: highest non-GT source per GT column.
    if active_cols.any():
        for ti in torch.nonzero(active_cols, as_tuple=False).flatten().tolist():
            col = logits[:, ti]
            gt_rows = torch.nonzero(target[:, ti] > 0, as_tuple=False).flatten()
            if len(gt_rows) == 0:
                continue
            for si in gt_rows.tolist():
                scores = col.clone()
                scores[si] = -1e9
                ni = int(torch.argmax(scores).item())
                if (si, ni, ti) in seen:
                    continue
                if target[ni, ti] > 0:
                    continue
                diff = logits[si, ti] - logits[ni, ti]
                if float(diff.detach()) >= MARGIN:
                    continue
                rank_terms.append(ONLINE_HARDNEG_WEIGHT * F.softplus(MARGIN - diff))
                n_ok += int((diff.detach() > 0).item())
    if rank_terms:
        rank_loss = torch.stack(rank_terms).mean()
    else:
        rank_loss = logits.sum() * 0.0
    n_pairs = len(rank_terms)
    return rank_loss, bce_loss, n_pairs, n_ok


def train_seed(
    *,
    seed_name: str,
    weights_path: Path,
    windows: list[CachedWindow] | None,
    cache_fn,
    output_dir: Path,
    device: torch.device,
    epochs: int,
    lr: float,
    load_model,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(weights_path.parent / "config.json", output_dir / "config.json")
    model, _window_size, _downsample = load_model(weights_path, device)
    for param in model.unet.parameters():
        param.requires_grad = False
    for param in model.detect_head.parameters():
        param.requires_grad = False
    model.unet.eval()
    model.detect_head.eval()
    if windows is None:
        print(f"[{seed_name}] caching UNet features on capture nodes", flush=True)
        t0 = time.perf_counter()
        windows = cache_fn(model)
        print(
            f"[{seed_name}] cached {len(windows)} windows in {time.perf_counter() - t0:.1f}s",
            flush=True,
        )

    trainable = [p for p in model.transformer.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.01)
    history: list[dict[str, float]] = []
    best_loss = float("inf")
    best_path = output_dir / "edge_predictor_best.pth"
    last_path = output_dir / "edge_predictor_last.pth"

    rng = np.random.default_rng(42 if "split_0" in seed_name else 314159)
    for epoch in range(1, epochs + 1):
        model.transformer.train()
        order = rng.permutation(len(windows))
        tot_rank = tot_bce = tot_loss = 0.0
        n_pairs = n_ok = n_used = 0
        t0 = time.perf_counter()
        for idx in order:
            window = windows[int(idx)]
            feat_src = window.feat_src.unsqueeze(0).to(device)
            feat_tgt = window.feat_tgt.unsqueeze(0).to(device)
            coords_src = window.coords_src.unsqueeze(0).to(device)
            coords_tgt = window.coords_tgt.unsqueeze(0).to(device)
            pos_src = window.pos_src.unsqueeze(0).to(device)
            pos_tgt = window.pos_tgt.unsqueeze(0).to(device)
            mask_s = torch.ones(1, feat_src.shape[1], dtype=torch.bool, device=device)
            mask_t = torch.ones(1, feat_tgt.shape[1], dtype=torch.bool, device=device)
            logits = model.predict_edges(
                feat_src, feat_tgt, coords_src, coords_tgt, pos_src, pos_tgt, mask_s, mask_t
            )[0]
            rank_loss, bce_loss, k_pairs, k_ok = _bce_ranking_losses(logits, window, device)
            loss = RANK_LOSS_WEIGHT * rank_loss + BCE_LOSS_WEIGHT * bce_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            tot_rank += float(rank_loss.detach())
            tot_bce += float(bce_loss.detach())
            tot_loss += float(loss.detach())
            n_pairs += k_pairs
            n_ok += k_ok
            n_used += 1
        n_used = max(n_used, 1)
        row = {
            "epoch": epoch,
            "rank_loss": tot_rank / n_used,
            "bce_loss": tot_bce / n_used,
            "loss": tot_loss / n_used,
            "pair_gt_win_rate": n_ok / max(n_pairs, 1),
            "n_rank_terms": n_pairs,
            "seconds": time.perf_counter() - t0,
        }
        history.append(row)
        print(f"[{seed_name}] epoch {epoch}/{epochs} {json.dumps(row)}", flush=True)
        torch.save(model.state_dict(), last_path)
        if row["loss"] < best_loss:
            best_loss = row["loss"]
            torch.save(model.state_dict(), best_path)
            print(f"[{seed_name}] saved best loss={best_loss:.6f}", flush=True)

    stats = {
        "seed_name": seed_name,
        "n_windows": len(windows),
        "epochs": epochs,
        "lr": lr,
        "best_loss": best_loss,
        "history": history,
        "best_checkpoint": str(best_path),
        "source_checkpoint": str(weights_path),
    }
    (output_dir / "train_log.json").write_text(json.dumps(stats, indent=2) + "\n")
    return stats


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--support-dir", type=Path, required=True)
    p.add_argument("--capture-root", type=Path, required=True)
    p.add_argument("--diagnostic-csv", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--lr", type=float, default=3e-5)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    _add_support_paths(args.support_dir)
    from predict_unet_transformer import load_model
    from train_unet_transformer import extract_pos_features

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("edge-scorer hard-neg retrain requires CUDA")
    print(f"device={device} {torch.cuda.get_device_name(0)}", flush=True)

    diagnostic = pd.read_csv(args.diagnostic_csv)
    pairs = mine_pairs_from_capture(diagnostic, args.capture_root, datasets=FIXED8_DATASETS)
    if pairs.empty:
        raise RuntimeError("no hard-neg pairs mined from fixed-8 capture")
    summary = summarize_pairs(pairs)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(args.output_dir / "mined_pairs.csv", index=False)
    (args.output_dir / "mine_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("MINE", json.dumps(summary), flush=True)

    seeds = [
        (
            "hardneg_v1_split_0",
            args.support_dir / "weights/unet_transformer/split_0/edge_predictor_best.pth",
        ),
        (
            "hardneg_v1_seed_314159",
            args.support_dir
            / "weights/unet_transformer/seed_314159/edge_predictor_best.pth",
        ),
    ]
    all_stats: dict[str, Any] = {"mine": summary, "seeds": []}
    for seed_name, ckpt in seeds:
        out = args.output_dir / seed_name

        def _cache(model, _seed=seed_name):
            windows: list[CachedWindow] = []
            for dataset in FIXED8_DATASETS:
                print(f"[{_seed}] cache {dataset}", flush=True)
                windows.extend(
                    cache_dataset_windows(
                        model=model,
                        dataset=dataset,
                        data_dir=args.data_dir,
                        capture_ds=args.capture_root / dataset,
                        pairs=pairs,
                        device=device,
                        extract_pos=extract_pos_features,
                    )
                )
            return windows

        stats = train_seed(
            seed_name=seed_name,
            weights_path=ckpt,
            windows=None,
            cache_fn=_cache,
            output_dir=out,
            device=device,
            epochs=args.epochs,
            lr=args.lr,
            load_model=load_model,
        )
        all_stats["seeds"].append(stats)
        dest = args.support_dir / "weights" / "unet_transformer" / seed_name
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copy2(out / "edge_predictor_best.pth", dest / "edge_predictor_best.pth")
        shutil.copy2(out / "config.json", dest / "config.json")
        print(f"installed {dest}", flush=True)
        del stats
        torch.cuda.empty_cache()

    (args.output_dir / "train_summary.json").write_text(json.dumps(all_stats, indent=2) + "\n")
    print("TRAIN_DONE", flush=True)


if __name__ == "__main__":
    main()
