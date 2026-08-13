"""External learned-detector/edge-scorer orchestration from the V106 notebook."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from biohub_pipeline.config import PipelineConfig


def validate_ensemble_alpha(alpha: float) -> None:
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("ensemble_alpha must be between 0 and 1")


def blend_logits(seed1: Any, seed2: Any, alpha: float) -> Any:
    """Blend compatible raw-logit arrays/tensors before any activation or threshold."""
    validate_ensemble_alpha(alpha)
    shape1 = getattr(seed1, "shape", None)
    shape2 = getattr(seed2, "shape", None)
    if shape1 is not None and shape2 is not None and tuple(shape1) != tuple(shape2):
        raise ValueError(f"cannot blend incompatible logit shapes: {shape1} != {shape2}")
    return alpha * seed1 + (1.0 - alpha) * seed2


def _checkpoint_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_ensemble_checkpoints(primary: Path, secondary: Path, alpha: float) -> None:
    validate_ensemble_alpha(alpha)
    if not secondary.is_file():
        raise FileNotFoundError(f"ensemble checkpoint is missing: {secondary}")
    if not primary.is_file():
        raise FileNotFoundError(f"primary checkpoint is missing: {primary}")
    if primary.resolve() == secondary.resolve() or _checkpoint_sha256(
        primary
    ) == _checkpoint_sha256(secondary):
        raise ValueError("ensemble requires two distinct independently trained checkpoints")


def list_stems(data_dir: Path) -> list[str]:
    stems = sorted(path.name[:-5] for path in data_dir.iterdir() if path.name.endswith(".zarr"))
    if not stems:
        raise FileNotFoundError(f"no test .zarr stores found in {data_dir}")
    return stems


def apply_spatial_d4_patch(repo_dir: Path, prediction_script: str) -> bool:
    """Apply the notebook's exact 4-way-to-D4 detector TTA source patch."""
    path = repo_dir / prediction_script
    source = path.read_text(encoding="utf-8")
    old = """        if cfg.det_tta:
            tta_flips = [(-1,), (-2,), (-2, -1)]
            for dims in tta_flips:
                imgs_flip = imgs.flip(dims)
                _, det_flip = model.encode(imgs_flip)
                for f in range(W):
                    det_logits[f] = det_logits[f] + det_flip[f].flip(dims)
                del imgs_flip, det_flip
            for f in range(W):
                det_logits[f] = det_logits[f] / 4"""
    new = """        if cfg.det_tta:
            _nv = 1
            for dims in [(-1,), (-2,), (-2, -1)]:
                imgs_flip = imgs.flip(dims)
                _, det_flip = model.encode(imgs_flip)
                for f in range(W):
                    det_logits[f] = det_logits[f] + det_flip[f].flip(dims)
                del imgs_flip, det_flip
                _nv += 1
            for _k in (1, 3):
                imgs_rot = torch.rot90(imgs, _k, dims=(-2, -1))
                _, det_rot = model.encode(imgs_rot)
                for f in range(W):
                    det_logits[f] = det_logits[f] + torch.rot90(det_rot[f], -_k, dims=(-2, -1))
                del imgs_rot, det_rot
                _nv += 1
            imgs_t = imgs.transpose(-1, -2)
            _, det_t = model.encode(imgs_t)
            for f in range(W):
                det_logits[f] = det_logits[f] + det_t[f].transpose(-1, -2)
            del imgs_t, det_t
            _nv += 1
            imgs_at = torch.rot90(imgs, 1, dims=(-2, -1)).transpose(-1, -2)
            _, det_at = model.encode(imgs_at)
            for f in range(W):
                det_logits[f] = det_logits[f] + torch.rot90(det_at[f].transpose(-1, -2), -1, dims=(-2, -1))
            del imgs_at, det_at
            _nv += 1
            for f in range(W):
                det_logits[f] = det_logits[f] / _nv"""
    if old in source:
        path.write_text(source.replace(old, new), encoding="utf-8")
        return True
    if new in source:
        return False
    raise RuntimeError("upstream detector TTA block does not match the V106 patch preimage")


def apply_logit_ensemble_patch(repo_dir: Path, prediction_script: str) -> bool:
    """Patch the support predictor with an opt-in raw detector/edge-logit blend wrapper."""
    path = repo_dir / prediction_script
    source = path.read_text(encoding="utf-8")
    if "_V106_LOGIT_ENSEMBLE_PATCH = True" in source:
        return False

    replacements = [
        (
            "import contextlib\n",
            "import contextlib\nimport hashlib\n",
        ),
        (
            "def load_model(\n    weights_path: Path, device: torch.device,\n) -> tuple[UNetNodeTransformer, int, tuple[int, ...]]:\n",
            '''_V106_LOGIT_ENSEMBLE_PATCH = True


def _blend_raw_logits(seed1: torch.Tensor, seed2: torch.Tensor, alpha: float) -> torch.Tensor:
    """Blend raw compatible logits before sigmoid/softmax or thresholding."""
    if seed1.shape != seed2.shape:
        raise ValueError(f"incompatible ensemble logit shapes: {seed1.shape} != {seed2.shape}")
    return alpha * seed1 + (1.0 - alpha) * seed2


class _EnsembleFeatures:
    """Keep model-specific feature tensors aligned through existing indexing code."""

    def __init__(self, seed1, seed2):
        self.seed1 = seed1
        self.seed2 = seed2

    def __getitem__(self, key):
        return _EnsembleFeatures(self.seed1[key], self.seed2[key])


class _LogitBlendModel:
    """Duck-typed predictor that blends both learned heads at their raw logits."""

    def __init__(self, seed1, seed2, alpha: float):
        self.seed1 = seed1
        self.seed2 = seed2
        self.alpha = alpha

    def encode(self, imgs):
        features1, detector1 = self.seed1.encode(imgs)
        features2, detector2 = self.seed2.encode(imgs)
        if len(detector1) != len(detector2):
            raise ValueError("incompatible ensemble detector output counts")
        detector = [
            _blend_raw_logits(logits1, logits2, self.alpha)
            for logits1, logits2 in zip(detector1, detector2)
        ]
        return _EnsembleFeatures(features1, features2), detector

    def _index_features(self, features, *args, **kwargs):
        return _EnsembleFeatures(
            self.seed1._index_features(features.seed1, *args, **kwargs),
            self.seed2._index_features(features.seed2, *args, **kwargs),
        )

    def predict_edges(self, source, target, *args, **kwargs):
        logits1 = self.seed1.predict_edges(source.seed1, target.seed1, *args, **kwargs)
        logits2 = self.seed2.predict_edges(source.seed2, target.seed2, *args, **kwargs)
        return _blend_raw_logits(logits1, logits2, self.alpha)


def _checkpoint_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_model(
    weights_path: Path, device: torch.device,
) -> tuple[UNetNodeTransformer, int, tuple[int, ...]]:
''',
        ),
        (
            "\n\n# =============================================================================\n# Per-frame loading\n# =============================================================================\n",
            """

def load_logit_ensemble(
    primary_path: Path,
    secondary_path: Path,
    alpha: float,
    device: torch.device,
):
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("ensemble alpha must be between 0 and 1")
    if not secondary_path.is_file():
        raise FileNotFoundError(f"ensemble checkpoint is missing: {secondary_path}")
    if primary_path.resolve() == secondary_path.resolve() or _checkpoint_sha256(primary_path) == _checkpoint_sha256(secondary_path):
        raise ValueError("ensemble requires two distinct independently trained checkpoints")

    seed1, window_size, downsample = load_model(primary_path, device)
    seed2, window_size2, downsample2 = load_model(secondary_path, device)
    signature1 = [(name, tuple(value.shape)) for name, value in seed1.state_dict().items()]
    signature2 = [(name, tuple(value.shape)) for name, value in seed2.state_dict().items()]
    if window_size != window_size2 or downsample != downsample2 or signature1 != signature2:
        raise ValueError("ensemble checkpoints are architecture-incompatible")
    return _LogitBlendModel(seed1, seed2, alpha), window_size, downsample


# =============================================================================
# Per-frame loading
# =============================================================================
""",
        ),
        (
            "    weights_path: Path,\n    cfg: PredictConfig,\n",
            "    weights_path: Path,\n    cfg: PredictConfig,\n    ensemble_weights_path: Path | None = None,\n    ensemble_alpha: float = 0.5,\n",
        ),
        (
            "    model, window_size, downsample = load_model(weights_path, device)\n",
            """    if ensemble_weights_path is None:
        model, window_size, downsample = load_model(weights_path, device)
    else:
        model, window_size, downsample = load_logit_ensemble(
            weights_path, ensemble_weights_path, ensemble_alpha, device,
        )
        print(
            f"Raw-logit ensemble: secondary={ensemble_weights_path} alpha={ensemble_alpha}",
            flush=True,
        )
""",
        ),
        (
            """    parser.add_argument("--weights", type=str, default=None,
                        help="Path to weights file. "
                             "Default: weights/{method}/split_{split}/edge_predictor_best.pth")
""",
            """    parser.add_argument("--weights", type=str, default=None,
                        help="Path to weights file. "
                             "Default: weights/{method}/split_{split}/edge_predictor_best.pth")
    parser.add_argument("--ensemble-weights", type=str, default=None,
                        help="Optional independent compatible checkpoint for raw-logit blending.")
    parser.add_argument("--ensemble-alpha", type=float, default=0.5,
                        help="Primary-checkpoint raw-logit weight in [0, 1] (default: 0.5).")
""",
        ),
        (
            "            weights_path=weights_path,\n            cfg=cfg,\n",
            """            weights_path=weights_path,
            cfg=cfg,
            ensemble_weights_path=(
                Path(args.ensemble_weights) if args.ensemble_weights else None
            ),
            ensemble_alpha=args.ensemble_alpha,
""",
        ),
    ]

    patched = source
    for old, new in replacements:
        if patched.count(old) != 1:
            raise RuntimeError("support predictor does not match the V106 ensemble patch preimage")
        patched = patched.replace(old, new, 1)
    path.write_text(patched, encoding="utf-8")
    return True


def apply_edge_diagnostic_patch(repo_dir: Path, prediction_script: str) -> bool:
    """Add opt-in pre-gate edge-score capture without changing graph construction."""
    path = repo_dir / prediction_script
    source = path.read_text(encoding="utf-8")
    if "_V106_EDGE_DIAGNOSTIC_PATCH = True" in source:
        return False
    if "_V106_LOGIT_ENSEMBLE_PATCH = True" not in source:
        raise RuntimeError("edge diagnostics require the logit-ensemble support patch first")

    replacements = [
        (
            "_V106_LOGIT_ENSEMBLE_PATCH = True\n",
            '''_V106_LOGIT_ENSEMBLE_PATCH = True
_V106_EDGE_DIAGNOSTIC_PATCH = True


def _write_edge_diagnostic(
    output_dir, t_src, t_tgt, idx_src, idx_tgt, raw, probs,
    threshold, top_k, seed_components,
):
    """Persist a bounded union of local top-k pairs plus every gated candidate."""
    output_dir.mkdir(parents=True, exist_ok=True)
    n_src, n_tgt = probs.shape
    k_src = min(int(top_k), n_tgt)
    k_tgt = min(int(top_k), n_src)
    source_top = torch.topk(probs, k_src, dim=1).indices
    target_top = torch.topk(probs, k_tgt, dim=0).indices
    source_pairs = torch.stack(
        [
            torch.arange(n_src, device=probs.device).repeat_interleave(k_src),
            source_top.reshape(-1),
        ],
        dim=1,
    )
    target_pairs = torch.stack(
        [
            target_top.transpose(0, 1).reshape(-1),
            torch.arange(n_tgt, device=probs.device).repeat_interleave(k_tgt),
        ],
        dim=1,
    )
    gated_pairs = torch.nonzero(probs > threshold, as_tuple=False)
    pairs = torch.unique(torch.cat([source_pairs, target_pairs, gated_pairs]), dim=0)

    pairs_cpu = pairs.cpu().numpy().astype(np.int32, copy=False)
    source_rank_lookup = {
        (int(i), int(j)): rank + 1
        for i, targets in enumerate(source_top.cpu().numpy())
        for rank, j in enumerate(targets)
    }
    target_rank_lookup = {
        (int(i), int(j)): rank + 1
        for j, sources in enumerate(target_top.transpose(0, 1).cpu().numpy())
        for rank, i in enumerate(sources)
    }
    local_source = pairs_cpu[:, 0]
    local_target = pairs_cpu[:, 1]
    source_rank = np.array(
        [source_rank_lookup.get((int(i), int(j)), k_src + 1) for i, j in pairs_cpu],
        dtype=np.int16,
    )
    target_rank = np.array(
        [target_rank_lookup.get((int(i), int(j)), k_tgt + 1) for i, j in pairs_cpu],
        dtype=np.int16,
    )
    pair_index = (pairs[:, 0], pairs[:, 1])
    blended_logit = raw[pair_index].float().cpu().numpy()
    blended_prob = probs[pair_index].float().cpu().numpy()
    above_threshold = (probs[pair_index] > threshold).cpu().numpy()

    seed1_logit = np.full(len(pairs_cpu), np.nan, dtype=np.float32)
    seed2_logit = np.full(len(pairs_cpu), np.nan, dtype=np.float32)
    seed1_prob = np.full(len(pairs_cpu), np.nan, dtype=np.float32)
    seed2_prob = np.full(len(pairs_cpu), np.nan, dtype=np.float32)
    if seed_components is not None:
        seed1, seed2 = seed_components
        seed1_values = seed1[pair_index].float()
        seed2_values = seed2[pair_index].float()
        seed1_logit = seed1_values.cpu().numpy()
        seed2_logit = seed2_values.cpu().numpy()
        seed1_prob = torch.exp(
            seed1_values - torch.logsumexp(seed1.float(), dim=0)[pairs[:, 1]]
        ).cpu().numpy()
        seed2_prob = torch.exp(
            seed2_values - torch.logsumexp(seed2.float(), dim=0)[pairs[:, 1]]
        ).cpu().numpy()

    np.savez_compressed(
        output_dir / f"t{int(t_src):03d}_to_t{int(t_tgt):03d}.npz",
        source_id=np.asarray(idx_src, dtype=np.int64)[local_source],
        target_id=np.asarray(idx_tgt, dtype=np.int64)[local_target],
        blended_logit=blended_logit,
        blended_prob=blended_prob,
        seed1_logit=seed1_logit,
        seed2_logit=seed2_logit,
        seed1_prob=seed1_prob,
        seed2_prob=seed2_prob,
        source_rank=source_rank,
        target_rank=target_rank,
        above_threshold=above_threshold,
    )
''',
        ),
        (
            """    def predict_edges(self, source, target, *args, **kwargs):
        logits1 = self.seed1.predict_edges(source.seed1, target.seed1, *args, **kwargs)
        logits2 = self.seed2.predict_edges(source.seed2, target.seed2, *args, **kwargs)
        return _blend_raw_logits(logits1, logits2, self.alpha)
""",
            """    def predict_edges(self, source, target, *args, **kwargs):
        logits1 = self.seed1.predict_edges(source.seed1, target.seed1, *args, **kwargs)
        logits2 = self.seed2.predict_edges(source.seed2, target.seed2, *args, **kwargs)
        blended = _blend_raw_logits(logits1, logits2, self.alpha)
        if getattr(self, "_capture_edge_diagnostics", False):
            self._last_edge_components = (logits1[0].detach(), logits2[0].detach())
        return blended
""",
        ),
        (
            """    unet_batch_size: int = 4,
    downsample: tuple[int, ...] = (1, 4, 4),
) -> tuple[np.ndarray, list[tuple[int, int, float, float]]]:
""",
            """    unet_batch_size: int = 4,
    downsample: tuple[int, ...] = (1, 4, 4),
    edge_diagnostic_dir: Path | None = None,
    edge_diagnostic_top_k: int = 16,
) -> tuple[np.ndarray, list[tuple[int, int, float, float]]]:
""",
        ),
        (
            """            if cfg.edge_activation == "softmax":
                probs = torch.softmax(raw, dim=0).cpu().numpy()
            else:
                probs = torch.sigmoid(raw).cpu().numpy()

            candidates = sorted(
""",
            """            if cfg.edge_activation == "softmax":
                probs_tensor = torch.softmax(raw, dim=0)
            else:
                probs_tensor = torch.sigmoid(raw)
            if edge_diagnostic_dir is not None:
                _write_edge_diagnostic(
                    edge_diagnostic_dir, t_src, t_tgt, idx_src, idx_tgt,
                    raw, probs_tensor, cfg.threshold, edge_diagnostic_top_k,
                    getattr(model, "_last_edge_components", None),
                )
                if hasattr(model, "_last_edge_components"):
                    del model._last_edge_components
            probs = probs_tensor.cpu().numpy()

            candidates = sorted(
""",
        ),
        (
            """    coords = coords.astype(np.int16)
    return coords, all_edges
""",
            """    coords = coords.astype(np.int16)
    if edge_diagnostic_dir is not None:
        edge_diagnostic_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            edge_diagnostic_dir / "nodes.npz",
            node_id=np.arange(len(coords), dtype=np.int64),
            t=coords[:, 0], z=coords[:, 1], y=coords[:, 2], x=coords[:, 3],
        )
        (edge_diagnostic_dir / "metadata.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "top_k": int(edge_diagnostic_top_k),
                    "edge_threshold": float(cfg.threshold),
                    "activation": cfg.edge_activation,
                    "pairs": len(seen_pairs),
                    "nodes": len(coords),
                },
                indent=2,
                sort_keys=True,
            ) + "\\n",
            encoding="utf-8",
        )
    return coords, all_edges
""",
        ),
        (
            """    video_slice: slice | None = None,
    evaluate: bool = False,
) -> None:
""",
            """    video_slice: slice | None = None,
    evaluate: bool = False,
    edge_diagnostic_dir: Path | None = None,
    edge_diagnostic_top_k: int = 16,
) -> None:
""",
        ),
        (
            """        print(
            f"Raw-logit ensemble: secondary={ensemble_weights_path} alpha={ensemble_alpha}",
            flush=True,
        )
""",
            """        print(
            f"Raw-logit ensemble: secondary={ensemble_weights_path} alpha={ensemble_alpha}",
            flush=True,
        )
    if edge_diagnostic_dir is not None:
        model._capture_edge_diagnostics = True
""",
        ),
        (
            """                unet_batch_size=unet_batch_size,
                downsample=downsample,
            )
""",
            """                unet_batch_size=unet_batch_size,
                downsample=downsample,
                edge_diagnostic_dir=(
                    edge_diagnostic_dir / name if edge_diagnostic_dir is not None else None
                ),
                edge_diagnostic_top_k=edge_diagnostic_top_k,
            )
""",
        ),
        (
            """    parser.add_argument("--evaluate", action="store_true",
                        help="Run evaluation against GT after saving predictions.")
""",
            """    parser.add_argument("--evaluate", action="store_true",
                        help="Run evaluation against GT after saving predictions.")
    parser.add_argument("--edge-diagnostic-dir", type=str, default=None,
                        help="Opt-in directory for bounded pre-gate edge-score exports.")
    parser.add_argument("--edge-diagnostic-top-k", type=int, default=16,
                        help="Top-k per source and target retained by diagnostics.")
""",
        ),
        (
            """            video_slice=video_slice,
            evaluate=args.evaluate,
        )
""",
            """            video_slice=video_slice,
            evaluate=args.evaluate,
            edge_diagnostic_dir=(
                Path(args.edge_diagnostic_dir) if args.edge_diagnostic_dir else None
            ),
            edge_diagnostic_top_k=args.edge_diagnostic_top_k,
        )
""",
        ),
    ]

    patched = source
    for old, new in replacements:
        if patched.count(old) != 1:
            raise RuntimeError("support predictor does not match edge diagnostic patch preimage")
        patched = patched.replace(old, new, 1)
    path.write_text(patched, encoding="utf-8")
    return True


def apply_edge_threshold_cli_patch(repo_dir: Path, prediction_script: str) -> bool:
    """Expose PredictConfig.threshold via an optional --edge-threshold CLI flag."""
    path = repo_dir / prediction_script
    source = path.read_text(encoding="utf-8")
    replacements = (
        (
            """    parser.add_argument("--det-threshold", type=float, default=0.99,
                        help="Min sigmoid probability for a detection peak to be kept. "
                             "Default 0.99: the detector is poorly calibrated because the "
                             "ground truth is sparse (only some cells annotated), so a high "
                             "threshold keeps precision up. Sweep it for your model.")
""",
            """    parser.add_argument("--det-threshold", type=float, default=0.99,
                        help="Min sigmoid probability for a detection peak to be kept. "
                             "Default 0.99: the detector is poorly calibrated because the "
                             "ground truth is sparse (only some cells annotated), so a high "
                             "threshold keeps precision up. Sweep it for your model.")
    parser.add_argument("--edge-threshold", type=float, default=0.5,
                        help="Min edge probability admitted to the linker (default: 0.5).")
""",
        ),
        (
            """    cfg = PredictConfig(
        det_threshold=args.det_threshold,
        use_ilp=args.use_ilp,
        ilp_edge_weight=args.ilp_edge_weight,
        ilp_appearance_weight=args.ilp_appearance_weight,
        ilp_disappearance_weight=args.ilp_disappearance_weight,
        ilp_division_weight=args.ilp_division_weight,
    )
""",
            """    cfg = PredictConfig(
        det_threshold=args.det_threshold,
        threshold=args.edge_threshold,
        use_ilp=args.use_ilp,
        ilp_edge_weight=args.ilp_edge_weight,
        ilp_appearance_weight=args.ilp_appearance_weight,
        ilp_disappearance_weight=args.ilp_disappearance_weight,
        ilp_division_weight=args.ilp_division_weight,
    )
""",
        ),
    )
    patched = source
    for old, new in replacements:
        if patched.count(old) != 1:
            raise RuntimeError("support predictor does not match edge-threshold patch preimage")
        patched = patched.replace(old, new, 1)
    path.write_text(patched, encoding="utf-8")
    return True


def apply_margin_gated_distance_rank_patch(repo_dir: Path, prediction_script: str) -> bool:
    """Opt-in pre-softmax distance boost for near-tie columns in dense scenes."""
    path = repo_dir / prediction_script
    source = path.read_text(encoding="utf-8")
    if "_V106_MARGIN_GATED_DISTANCE_RANK_PATCH = True" in source:
        return False

    marker = "@dataclass\nclass PredictConfig:\n"
    if source.count(marker) != 1:
        raise RuntimeError("support predictor missing PredictConfig for distance-rank patch")

    helpers = '''_V106_MARGIN_GATED_DISTANCE_RANK_PATCH = True


def _margin_gated_distance_adjust(
    raw, coords_so_far, idx_src, idx_tgt, voxel_size,
    lam, delta, dens_min, radius_um,
):
    """Boost logits by lam*dist_um for near-tie target columns in dense scenes."""
    import numpy as _np
    import torch as _torch
    was_tensor = isinstance(raw, _torch.Tensor)
    device = raw.device if was_tensor else None
    logits = raw.detach().float().cpu().numpy() if was_tensor else _np.asarray(raw, dtype=_np.float64)
    n_src, n_tgt = logits.shape
    if n_src < 2 or n_tgt < 1:
        return raw
    src_zyx = _np.asarray(coords_so_far, dtype=_np.float64)[_np.asarray(idx_src, dtype=_np.int64)][:, 1:4]
    tgt_zyx = _np.asarray(coords_so_far, dtype=_np.float64)[_np.asarray(idx_tgt, dtype=_np.int64)][:, 1:4]
    scale = _np.asarray(voxel_size, dtype=_np.float64).reshape(1, 1, 3)
    dists = _np.linalg.norm(
        (src_zyx[:, None, :] - tgt_zyx[None, :, :]) * scale, axis=2
    )
    tgt_scale = _np.asarray(voxel_size, dtype=_np.float64).reshape(1, 1, 3)
    d_tgt = _np.linalg.norm(
        (tgt_zyx[:, None, :] - tgt_zyx[None, :, :]) * tgt_scale, axis=2
    )
    dens = (d_tgt <= float(radius_um)).sum(axis=1)
    out = logits.copy()
    for j in range(n_tgt):
        col = out[:, j]
        top2 = _np.partition(col, -2)[-2:]
        gap = float(top2.max() - top2.min())
        if gap < float(delta) and float(dens[j]) >= float(dens_min):
            out[:, j] = col + float(lam) * dists[:, j]
    if was_tensor:
        return _torch.from_numpy(out).to(device=device, dtype=raw.dtype)
    return out


'''
    # Insert helper immediately before @dataclass PredictConfig
    source = source.replace(marker, helpers + marker, 1)

    replacements = [
        (
            """    # Edge filtering
    edge_activation: str = "softmax"  # "sigmoid" or "softmax"
    threshold: float = 0.5
""",
            """    # Edge filtering
    edge_activation: str = "softmax"  # "sigmoid" or "softmax"
    threshold: float = 0.5
    # Opt-in margin-gated distance rank (pre-softmax). lam=0 disables.
    margin_gated_dist_lambda: float = 0.0
    margin_gated_dist_delta: float = 0.15
    margin_gated_dist_dens_min: float = 8.0
    margin_gated_dist_radius_um: float = 15.0
""",
        ),
        (
            """            raw = edge_logits_pair[0]
            if cfg.edge_activation == "softmax":
""",
            """            raw = edge_logits_pair[0]
            if float(getattr(cfg, "margin_gated_dist_lambda", 0.0) or 0.0) != 0.0:
                raw = _margin_gated_distance_adjust(
                    raw, coords_so_far, idx_src, idx_tgt, voxel_size,
                    float(cfg.margin_gated_dist_lambda),
                    float(cfg.margin_gated_dist_delta),
                    float(cfg.margin_gated_dist_dens_min),
                    float(cfg.margin_gated_dist_radius_um),
                )
            if cfg.edge_activation == "softmax":
""",
        ),
        (
            """    parser.add_argument("--edge-threshold", type=float, default=0.5,
                        help="Min edge probability admitted to the linker (default: 0.5).")
""",
            """    parser.add_argument("--edge-threshold", type=float, default=0.5,
                        help="Min edge probability admitted to the linker (default: 0.5).")
    parser.add_argument("--margin-gated-dist-lambda", type=float, default=0.0,
                        help="Opt-in pre-softmax distance bonus for near-tie dense columns.")
    parser.add_argument("--margin-gated-dist-delta", type=float, default=0.15,
                        help="Logit top1-top2 gap below which distance bonus applies.")
    parser.add_argument("--margin-gated-dist-dens-min", type=float, default=8.0,
                        help="Min local detection density (radius_um) to apply bonus.")
    parser.add_argument("--margin-gated-dist-radius-um", type=float, default=15.0,
                        help="Density neighborhood radius in µm.")
""",
        ),
        (
            """    cfg = PredictConfig(
        det_threshold=args.det_threshold,
        threshold=args.edge_threshold,
        use_ilp=args.use_ilp,
        ilp_edge_weight=args.ilp_edge_weight,
        ilp_appearance_weight=args.ilp_appearance_weight,
        ilp_disappearance_weight=args.ilp_disappearance_weight,
        ilp_division_weight=args.ilp_division_weight,
    )
""",
            """    cfg = PredictConfig(
        det_threshold=args.det_threshold,
        threshold=args.edge_threshold,
        use_ilp=args.use_ilp,
        ilp_edge_weight=args.ilp_edge_weight,
        ilp_appearance_weight=args.ilp_appearance_weight,
        ilp_disappearance_weight=args.ilp_disappearance_weight,
        ilp_division_weight=args.ilp_division_weight,
        margin_gated_dist_lambda=args.margin_gated_dist_lambda,
        margin_gated_dist_delta=args.margin_gated_dist_delta,
        margin_gated_dist_dens_min=args.margin_gated_dist_dens_min,
        margin_gated_dist_radius_um=args.margin_gated_dist_radius_um,
    )
""",
        ),
    ]

    # Edge-threshold patch may not have been applied yet; also accept unpatched PredictConfig block.
    alt_cfg = (
        """    cfg = PredictConfig(
        det_threshold=args.det_threshold,
        use_ilp=args.use_ilp,
        ilp_edge_weight=args.ilp_edge_weight,
        ilp_appearance_weight=args.ilp_appearance_weight,
        ilp_disappearance_weight=args.ilp_disappearance_weight,
        ilp_division_weight=args.ilp_division_weight,
    )
""",
        """    cfg = PredictConfig(
        det_threshold=args.det_threshold,
        use_ilp=args.use_ilp,
        ilp_edge_weight=args.ilp_edge_weight,
        ilp_appearance_weight=args.ilp_appearance_weight,
        ilp_disappearance_weight=args.ilp_disappearance_weight,
        ilp_division_weight=args.ilp_division_weight,
        margin_gated_dist_lambda=args.margin_gated_dist_lambda,
        margin_gated_dist_delta=args.margin_gated_dist_delta,
        margin_gated_dist_dens_min=args.margin_gated_dist_dens_min,
        margin_gated_dist_radius_um=args.margin_gated_dist_radius_um,
    )
""",
    )
    alt_cli = (
        """    parser.add_argument("--det-threshold", type=float, default=0.99,
                        help="Min sigmoid probability for a detection peak to be kept. "
                             "Default 0.99: the detector is poorly calibrated because the "
                             "ground truth is sparse (only some cells annotated), so a high "
                             "threshold keeps precision up. Sweep it for your model.")
""",
        """    parser.add_argument("--det-threshold", type=float, default=0.99,
                        help="Min sigmoid probability for a detection peak to be kept. "
                             "Default 0.99: the detector is poorly calibrated because the "
                             "ground truth is sparse (only some cells annotated), so a high "
                             "threshold keeps precision up. Sweep it for your model.")
    parser.add_argument("--margin-gated-dist-lambda", type=float, default=0.0,
                        help="Opt-in pre-softmax distance bonus for near-tie dense columns.")
    parser.add_argument("--margin-gated-dist-delta", type=float, default=0.15,
                        help="Logit top1-top2 gap below which distance bonus applies.")
    parser.add_argument("--margin-gated-dist-dens-min", type=float, default=8.0,
                        help="Min local detection density (radius_um) to apply bonus.")
    parser.add_argument("--margin-gated-dist-radius-um", type=float, default=15.0,
                        help="Density neighborhood radius in µm.")
""",
    )

    patched = source
    for old, new in replacements:
        if patched.count(old) == 1:
            patched = patched.replace(old, new, 1)
        elif old == replacements[2][0] and patched.count(alt_cli[0]) == 1:
            patched = patched.replace(alt_cli[0], alt_cli[1], 1)
        elif old == replacements[3][0] and patched.count(alt_cfg[0]) == 1:
            patched = patched.replace(alt_cfg[0], alt_cfg[1], 1)
        else:
            raise RuntimeError(
                "support predictor does not match margin-gated distance-rank patch preimage"
            )
    path.write_text(patched, encoding="utf-8")
    return True


def apply_pairwise_hardneg_rank_patch(repo_dir: Path, prediction_script: str) -> bool:
    """Opt-in pre-softmax appearance-aware pairwise hard-neg reweight."""
    path = repo_dir / prediction_script
    source = path.read_text(encoding="utf-8")
    if "_V106_PAIRWISE_HARDNEG_RANK_PATCH = True" in source:
        return False

    marker = "@dataclass\nclass PredictConfig:\n"
    if source.count(marker) != 1:
        raise RuntimeError("support predictor missing PredictConfig for pairwise hardneg patch")

    helpers = '''_V106_PAIRWISE_HARDNEG_RANK_PATCH = True


def _pairwise_hardneg_adjust(
    raw, seed1, seed2, coords_so_far, idx_src, idx_tgt, voxel_size,
    weights, dens_min, gap_max, radius_um,
):
    """blended_logit + w·φ(seed_disagree, seed_min, dist, dens) on near-tie dense columns."""
    import json as _json
    import numpy as _np
    import torch as _torch
    was_tensor = isinstance(raw, _torch.Tensor)
    device = raw.device if was_tensor else None
    logits = raw.detach().float().cpu().numpy() if was_tensor else _np.asarray(raw, dtype=_np.float64)
    s1 = seed1.detach().float().cpu().numpy() if isinstance(seed1, _torch.Tensor) else _np.asarray(seed1, dtype=_np.float64)
    s2 = seed2.detach().float().cpu().numpy() if isinstance(seed2, _torch.Tensor) else _np.asarray(seed2, dtype=_np.float64)
    w = _np.asarray(weights if not isinstance(weights, str) else _json.loads(weights), dtype=_np.float64).reshape(-1)
    n_src, n_tgt = logits.shape
    if n_src < 2 or n_tgt < 1 or w.shape[0] != 6:
        return raw
    src_zyx = _np.asarray(coords_so_far, dtype=_np.float64)[_np.asarray(idx_src, dtype=_np.int64)][:, 1:4]
    tgt_zyx = _np.asarray(coords_so_far, dtype=_np.float64)[_np.asarray(idx_tgt, dtype=_np.int64)][:, 1:4]
    scale = _np.asarray(voxel_size, dtype=_np.float64).reshape(1, 1, 3)
    dists = _np.linalg.norm((src_zyx[:, None, :] - tgt_zyx[None, :, :]) * scale, axis=2)
    d_tgt = _np.linalg.norm((tgt_zyx[:, None, :] - tgt_zyx[None, :, :]) * scale, axis=2)
    dens = (d_tgt <= float(radius_um)).sum(axis=1)
    disagree = _np.abs(s1 - s2)
    smin = _np.minimum(s1, s2)
    dist_n = dists / 10.0
    log_dens = _np.log1p(_np.maximum(dens.astype(_np.float64), 0.0))
    log_dens_b = _np.broadcast_to(log_dens.reshape(1, -1), dist_n.shape)
    feats = _np.stack(
        [disagree, smin, dist_n, log_dens_b, dist_n * disagree, dist_n * smin],
        axis=-1,
    )
    correction = feats @ w
    out = logits.copy()
    for j in range(n_tgt):
        col = out[:, j]
        top2 = _np.partition(col, -2)[-2:]
        gap = float(top2.max() - top2.min())
        if gap < float(gap_max) and float(dens[j]) >= float(dens_min):
            out[:, j] = col + correction[:, j]
    if was_tensor:
        return _torch.from_numpy(out).to(device=device, dtype=raw.dtype)
    return out


'''
    source = source.replace(marker, helpers + marker, 1)

    replacements = [
        (
            """    # Edge filtering
    edge_activation: str = "softmax"  # "sigmoid" or "softmax"
    threshold: float = 0.5
""",
            """    # Edge filtering
    edge_activation: str = "softmax"  # "sigmoid" or "softmax"
    threshold: float = 0.5
    # Opt-in pairwise hard-neg reweight (pre-softmax). Empty weights disables.
    pairwise_hardneg_weights: str = ""
    pairwise_hardneg_dens_min: float = 8.0
    pairwise_hardneg_gap_max: float = 1.5
    pairwise_hardneg_radius_um: float = 15.0
""",
        ),
        (
            """    def predict_edges(self, source, target, *args, **kwargs):
        logits1 = self.seed1.predict_edges(source.seed1, target.seed1, *args, **kwargs)
        logits2 = self.seed2.predict_edges(source.seed2, target.seed2, *args, **kwargs)
        return _blend_raw_logits(logits1, logits2, self.alpha)
""",
            """    def predict_edges(self, source, target, *args, **kwargs):
        logits1 = self.seed1.predict_edges(source.seed1, target.seed1, *args, **kwargs)
        logits2 = self.seed2.predict_edges(source.seed2, target.seed2, *args, **kwargs)
        blended = _blend_raw_logits(logits1, logits2, self.alpha)
        self._last_edge_components = (logits1[0].detach(), logits2[0].detach())
        return blended
""",
        ),
        (
            """            raw = edge_logits_pair[0]
            if cfg.edge_activation == "softmax":
""",
            """            raw = edge_logits_pair[0]
            _phw = getattr(cfg, "pairwise_hardneg_weights", "") or ""
            if _phw:
                _comps = getattr(model, "_last_edge_components", None)
                if _comps is not None:
                    raw = _pairwise_hardneg_adjust(
                        raw, _comps[0], _comps[1], coords_so_far, idx_src, idx_tgt, voxel_size,
                        _phw,
                        float(cfg.pairwise_hardneg_dens_min),
                        float(cfg.pairwise_hardneg_gap_max),
                        float(cfg.pairwise_hardneg_radius_um),
                    )
            if cfg.edge_activation == "softmax":
""",
        ),
        (
            """    parser.add_argument("--edge-threshold", type=float, default=0.5,
                        help="Min edge probability admitted to the linker (default: 0.5).")
""",
            """    parser.add_argument("--edge-threshold", type=float, default=0.5,
                        help="Min edge probability admitted to the linker (default: 0.5).")
    parser.add_argument("--pairwise-hardneg-weights", type=str, default="",
                        help="JSON list of 6 weights; empty disables pairwise hard-neg reweight.")
    parser.add_argument("--pairwise-hardneg-dens-min", type=float, default=8.0,
                        help="Min local detection density to apply pairwise reweight.")
    parser.add_argument("--pairwise-hardneg-gap-max", type=float, default=1.5,
                        help="Apply only when top1-top2 logit gap is below this.")
    parser.add_argument("--pairwise-hardneg-radius-um", type=float, default=15.0,
                        help="Density neighborhood radius in µm.")
""",
        ),
        (
            """    cfg = PredictConfig(
        det_threshold=args.det_threshold,
        threshold=args.edge_threshold,
        use_ilp=args.use_ilp,
        ilp_edge_weight=args.ilp_edge_weight,
        ilp_appearance_weight=args.ilp_appearance_weight,
        ilp_disappearance_weight=args.ilp_disappearance_weight,
        ilp_division_weight=args.ilp_division_weight,
    )
""",
            """    cfg = PredictConfig(
        det_threshold=args.det_threshold,
        threshold=args.edge_threshold,
        use_ilp=args.use_ilp,
        ilp_edge_weight=args.ilp_edge_weight,
        ilp_appearance_weight=args.ilp_appearance_weight,
        ilp_disappearance_weight=args.ilp_disappearance_weight,
        ilp_division_weight=args.ilp_division_weight,
        pairwise_hardneg_weights=args.pairwise_hardneg_weights,
        pairwise_hardneg_dens_min=args.pairwise_hardneg_dens_min,
        pairwise_hardneg_gap_max=args.pairwise_hardneg_gap_max,
        pairwise_hardneg_radius_um=args.pairwise_hardneg_radius_um,
    )
""",
        ),
    ]

    # Accept PredictConfig that already has margin-gated fields.
    alt_cfg_margin = (
        """    cfg = PredictConfig(
        det_threshold=args.det_threshold,
        threshold=args.edge_threshold,
        use_ilp=args.use_ilp,
        ilp_edge_weight=args.ilp_edge_weight,
        ilp_appearance_weight=args.ilp_appearance_weight,
        ilp_disappearance_weight=args.ilp_disappearance_weight,
        ilp_division_weight=args.ilp_division_weight,
        margin_gated_dist_lambda=args.margin_gated_dist_lambda,
        margin_gated_dist_delta=args.margin_gated_dist_delta,
        margin_gated_dist_dens_min=args.margin_gated_dist_dens_min,
        margin_gated_dist_radius_um=args.margin_gated_dist_radius_um,
    )
""",
        """    cfg = PredictConfig(
        det_threshold=args.det_threshold,
        threshold=args.edge_threshold,
        use_ilp=args.use_ilp,
        ilp_edge_weight=args.ilp_edge_weight,
        ilp_appearance_weight=args.ilp_appearance_weight,
        ilp_disappearance_weight=args.ilp_disappearance_weight,
        ilp_division_weight=args.ilp_division_weight,
        margin_gated_dist_lambda=args.margin_gated_dist_lambda,
        margin_gated_dist_delta=args.margin_gated_dist_delta,
        margin_gated_dist_dens_min=args.margin_gated_dist_dens_min,
        margin_gated_dist_radius_um=args.margin_gated_dist_radius_um,
        pairwise_hardneg_weights=args.pairwise_hardneg_weights,
        pairwise_hardneg_dens_min=args.pairwise_hardneg_dens_min,
        pairwise_hardneg_gap_max=args.pairwise_hardneg_gap_max,
        pairwise_hardneg_radius_um=args.pairwise_hardneg_radius_um,
    )
""",
    )
    alt_cfg_plain = (
        """    cfg = PredictConfig(
        det_threshold=args.det_threshold,
        use_ilp=args.use_ilp,
        ilp_edge_weight=args.ilp_edge_weight,
        ilp_appearance_weight=args.ilp_appearance_weight,
        ilp_disappearance_weight=args.ilp_disappearance_weight,
        ilp_division_weight=args.ilp_division_weight,
    )
""",
        """    cfg = PredictConfig(
        det_threshold=args.det_threshold,
        use_ilp=args.use_ilp,
        ilp_edge_weight=args.ilp_edge_weight,
        ilp_appearance_weight=args.ilp_appearance_weight,
        ilp_disappearance_weight=args.ilp_disappearance_weight,
        ilp_division_weight=args.ilp_division_weight,
        pairwise_hardneg_weights=args.pairwise_hardneg_weights,
        pairwise_hardneg_dens_min=args.pairwise_hardneg_dens_min,
        pairwise_hardneg_gap_max=args.pairwise_hardneg_gap_max,
        pairwise_hardneg_radius_um=args.pairwise_hardneg_radius_um,
    )
""",
    )
    alt_cli = (
        """    parser.add_argument("--det-threshold", type=float, default=0.99,
                        help="Min sigmoid probability for a detection peak to be kept. "
                             "Default 0.99: the detector is poorly calibrated because the "
                             "ground truth is sparse (only some cells annotated), so a high "
                             "threshold keeps precision up. Sweep it for your model.")
""",
        """    parser.add_argument("--det-threshold", type=float, default=0.99,
                        help="Min sigmoid probability for a detection peak to be kept. "
                             "Default 0.99: the detector is poorly calibrated because the "
                             "ground truth is sparse (only some cells annotated), so a high "
                             "threshold keeps precision up. Sweep it for your model.")
    parser.add_argument("--pairwise-hardneg-weights", type=str, default="",
                        help="JSON list of 6 weights; empty disables pairwise hard-neg reweight.")
    parser.add_argument("--pairwise-hardneg-dens-min", type=float, default=8.0,
                        help="Min local detection density to apply pairwise reweight.")
    parser.add_argument("--pairwise-hardneg-gap-max", type=float, default=1.5,
                        help="Apply only when top1-top2 logit gap is below this.")
    parser.add_argument("--pairwise-hardneg-radius-um", type=float, default=15.0,
                        help="Density neighborhood radius in µm.")
""",
    )
    # Diagnostic patch may already stash components conditionally.
    alt_predict = (
        """    def predict_edges(self, source, target, *args, **kwargs):
        logits1 = self.seed1.predict_edges(source.seed1, target.seed1, *args, **kwargs)
        logits2 = self.seed2.predict_edges(source.seed2, target.seed2, *args, **kwargs)
        blended = _blend_raw_logits(logits1, logits2, self.alpha)
        if getattr(self, "_capture_edge_diagnostics", False):
            self._last_edge_components = (logits1[0].detach(), logits2[0].detach())
        return blended
""",
        """    def predict_edges(self, source, target, *args, **kwargs):
        logits1 = self.seed1.predict_edges(source.seed1, target.seed1, *args, **kwargs)
        logits2 = self.seed2.predict_edges(source.seed2, target.seed2, *args, **kwargs)
        blended = _blend_raw_logits(logits1, logits2, self.alpha)
        self._last_edge_components = (logits1[0].detach(), logits2[0].detach())
        return blended
""",
    )
    # Margin-gated may already wrap raw=
    alt_raw = (
        """            raw = edge_logits_pair[0]
            if float(getattr(cfg, "margin_gated_dist_lambda", 0.0) or 0.0) != 0.0:
                raw = _margin_gated_distance_adjust(
                    raw, coords_so_far, idx_src, idx_tgt, voxel_size,
                    float(cfg.margin_gated_dist_lambda),
                    float(cfg.margin_gated_dist_delta),
                    float(cfg.margin_gated_dist_dens_min),
                    float(cfg.margin_gated_dist_radius_um),
                )
            if cfg.edge_activation == "softmax":
""",
        """            raw = edge_logits_pair[0]
            if float(getattr(cfg, "margin_gated_dist_lambda", 0.0) or 0.0) != 0.0:
                raw = _margin_gated_distance_adjust(
                    raw, coords_so_far, idx_src, idx_tgt, voxel_size,
                    float(cfg.margin_gated_dist_lambda),
                    float(cfg.margin_gated_dist_delta),
                    float(cfg.margin_gated_dist_dens_min),
                    float(cfg.margin_gated_dist_radius_um),
                )
            _phw = getattr(cfg, "pairwise_hardneg_weights", "") or ""
            if _phw:
                _comps = getattr(model, "_last_edge_components", None)
                if _comps is not None:
                    raw = _pairwise_hardneg_adjust(
                        raw, _comps[0], _comps[1], coords_so_far, idx_src, idx_tgt, voxel_size,
                        _phw,
                        float(cfg.pairwise_hardneg_dens_min),
                        float(cfg.pairwise_hardneg_gap_max),
                        float(cfg.pairwise_hardneg_radius_um),
                    )
            if cfg.edge_activation == "softmax":
""",
    )

    patched = source
    for old, new in replacements:
        if patched.count(old) == 1:
            patched = patched.replace(old, new, 1)
        elif old == replacements[1][0] and patched.count(alt_predict[0]) == 1:
            patched = patched.replace(alt_predict[0], alt_predict[1], 1)
        elif old == replacements[2][0] and patched.count(alt_raw[0]) == 1:
            patched = patched.replace(alt_raw[0], alt_raw[1], 1)
        elif old == replacements[3][0] and patched.count(alt_cli[0]) == 1:
            patched = patched.replace(alt_cli[0], alt_cli[1], 1)
        elif old == replacements[4][0] and patched.count(alt_cfg_margin[0]) == 1:
            patched = patched.replace(alt_cfg_margin[0], alt_cfg_margin[1], 1)
        elif old == replacements[4][0] and patched.count(alt_cfg_plain[0]) == 1:
            patched = patched.replace(alt_cfg_plain[0], alt_cfg_plain[1], 1)
        else:
            raise RuntimeError(
                "support predictor does not match pairwise hard-neg rank patch preimage"
            )
    path.write_text(patched, encoding="utf-8")
    return True


def resolve_ensemble_weights(config: PipelineConfig, primary_weights: Path) -> Path | None:
    relative = config.inference.get("ensemble_weights_relative")
    if relative is None:
        return None
    primary_relative = Path(str(config.inference["weights_relative"]))
    weights_root = primary_weights
    for _ in primary_relative.parts:
        weights_root = weights_root.parent
    return (weights_root / Path(str(relative))).resolve()


def build_predict_command(
    config: PipelineConfig,
    data_dir: Path,
    repo_dir: Path,
    weights_path: Path,
    stems: list[str],
) -> tuple[list[str], Path]:
    splits = repo_dir / "clean_v106_test_splits.json"
    splits.write_text(
        json.dumps([{"split": 0, "train": [], "test": stems}], indent=2), encoding="utf-8"
    )
    inf = config.inference
    relative_weights = os.path.relpath(weights_path, repo_dir)
    ensemble_weights = resolve_ensemble_weights(config, weights_path)
    if ensemble_weights is not None:
        alpha = float(inf.get("ensemble_alpha", 0.5))
        validate_ensemble_checkpoints(weights_path, ensemble_weights, alpha)
        apply_logit_ensemble_patch(repo_dir, str(inf["prediction_script"]))
    edge_threshold = inf.get("edge_threshold")
    if edge_threshold is not None:
        apply_edge_threshold_cli_patch(repo_dir, str(inf["prediction_script"]))
    dist_lam = float(inf.get("margin_gated_dist_lambda") or 0.0)
    if dist_lam != 0.0:
        apply_margin_gated_distance_rank_patch(repo_dir, str(inf["prediction_script"]))
    pairwise_w = inf.get("pairwise_hardneg_w")
    if pairwise_w:
        apply_pairwise_hardneg_rank_patch(repo_dir, str(inf["prediction_script"]))
    command = [
        sys.executable,
        str(inf["prediction_script"]),
        "--data-dir",
        str(data_dir),
        "--splits",
        splits.name,
        "--split",
        "0",
        "--weights",
        relative_weights,
        "--unet-batch-size",
        str(inf["unet_batch_size"]),
        "--det-threshold",
        str(inf["detection_threshold"]),
        "--ilp-edge-weight",
        str(inf["ilp_edge_weight"]),
        "--ilp-appearance-weight",
        str(inf["ilp_appearance_weight"]),
        "--ilp-disappearance-weight",
        str(inf["ilp_disappearance_weight"]),
        "--ilp-division-weight",
        str(inf["ilp_division_weight"]),
    ]
    if edge_threshold is not None:
        command.extend(["--edge-threshold", str(float(edge_threshold))])
    if ensemble_weights is not None:
        command.extend(
            [
                "--ensemble-weights",
                os.path.relpath(ensemble_weights, repo_dir),
                "--ensemble-alpha",
                str(inf.get("ensemble_alpha", 0.5)),
            ]
        )
    if dist_lam != 0.0:
        command.extend(
            [
                "--margin-gated-dist-lambda",
                str(dist_lam),
                "--margin-gated-dist-delta",
                str(float(inf.get("margin_gated_dist_delta", 0.15))),
                "--margin-gated-dist-dens-min",
                str(float(inf.get("margin_gated_dist_dens_min", 8.0))),
                "--margin-gated-dist-radius-um",
                str(float(inf.get("margin_gated_dist_radius_um", 15.0))),
            ]
        )
    if pairwise_w:
        command.extend(
            [
                "--pairwise-hardneg-weights",
                json.dumps([float(x) for x in pairwise_w]),
                "--pairwise-hardneg-dens-min",
                str(float(inf.get("pairwise_hardneg_dens_min", 8.0))),
                "--pairwise-hardneg-gap-max",
                str(float(inf.get("pairwise_hardneg_gap_max", 1.5))),
                "--pairwise-hardneg-radius-um",
                str(float(inf.get("pairwise_hardneg_radius_um", 15.0))),
            ]
        )
    if inf["use_ilp"]:
        command.append("--use-ilp")
    return command, splits


def run_prediction(command: list[str], repo_dir: Path) -> None:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; V106 refuses CPU inference to avoid timeout")
    subprocess.run(command, cwd=repo_dir, env={**os.environ, "PYTHONPATH": "src"}, check=True)
