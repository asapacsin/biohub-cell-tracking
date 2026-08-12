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
    if inf["use_ilp"]:
        command.append("--use-ilp")
    return command, splits


def run_prediction(command: list[str], repo_dir: Path) -> None:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; V106 refuses CPU inference to avoid timeout")
    subprocess.run(command, cwd=repo_dir, env={**os.environ, "PYTHONPATH": "src"}, check=True)
