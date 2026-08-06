from __future__ import annotations

import json
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from biohub_tracker.preprocessing import PreprocessingConfig
from biohub_tracker.training.data import (
    AugmentationConfig,
    CentroidPatchDataset,
    DatasetView,
    PatchMixConfig,
)
from biohub_tracker.training.samplers import FrameGroupedBatchSampler


def _torch() -> Any:
    try:
        import torch  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - optional runtime
        raise RuntimeError(
            "Detector training requires PyTorch; install with `pip install -e '.[ml]'`"
        ) from exc
    return torch


@dataclass(frozen=True, slots=True)
class UNet3DConfig:
    in_channels: int = 1
    out_channels: int = 1
    base_channels: int = 16
    depth: int = 3

    def __post_init__(self) -> None:
        if min(self.in_channels, self.out_channels, self.base_channels, self.depth) < 1:
            raise ValueError("U-Net channel counts and depth must be positive")


@dataclass(frozen=True, slots=True)
class DetectorTrainingConfig:
    patch_shape_zyx: tuple[int, int, int] = (32, 128, 128)
    sigma_um: float = 2.0
    jitter_voxels_zyx: tuple[int, int, int] = (2, 8, 8)
    epochs: int = 20
    batch_size: int = 2
    learning_rate: float = 1e-3
    validation_fraction: float = 0.1
    positive_weight: float = 20.0
    num_workers: int = 0
    seed: int = 42
    device: str = "cuda"
    frame_cache_size: int = 4
    patch_mix: PatchMixConfig = field(default_factory=PatchMixConfig)
    positive_center_radius_um: float = 6.0
    empty_exclusion_margin_um: float = 4.0
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    unet: UNet3DConfig = field(default_factory=UNet3DConfig)
    frame_grouped_batches: bool = True
    use_amp: bool = True
    resume: bool = True
    log_every: int = 200

    def __post_init__(self) -> None:
        divisor = 2**self.unet.depth
        if any(size < divisor or size % divisor for size in self.patch_shape_zyx):
            raise ValueError(f"patch dimensions must be divisible by 2**depth ({divisor})")
        if self.epochs < 1 or self.batch_size < 1 or self.learning_rate <= 0:
            raise ValueError("epochs/batch_size must be positive and learning_rate must be > 0")
        if not 0 <= self.validation_fraction < 1:
            raise ValueError("validation_fraction must be in [0, 1)")
        if self.positive_weight <= 0 or self.sigma_um <= 0 or self.num_workers < 0:
            raise ValueError("training weights/sigma must be positive and workers non-negative")
        if self.frame_cache_size < 1:
            raise ValueError("frame_cache_size must be positive")
        if self.positive_center_radius_um <= 0 or self.empty_exclusion_margin_um < 0:
            raise ValueError("positive_center_radius_um must be > 0 and margin >= 0")
        if self.log_every < 1:
            raise ValueError("log_every must be positive")


def build_unet3d(config: UNet3DConfig) -> Any:
    """Build a conventional encoder/decoder 3D U-Net without importing torch at package load."""
    torch = _torch()
    nn = torch.nn
    module_base: Any = nn.Module

    class DoubleConv(module_base):  # type: ignore[misc]
        def __init__(self, in_channels: int, out_channels: int) -> None:
            super().__init__()
            self.layers = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, 3, padding=1, bias=False),
                nn.InstanceNorm3d(out_channels, affine=True),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Conv3d(out_channels, out_channels, 3, padding=1, bias=False),
                nn.InstanceNorm3d(out_channels, affine=True),
                nn.LeakyReLU(0.1, inplace=True),
            )

        def forward(self, inputs: Any) -> Any:
            return self.layers(inputs)

    class UNet3D(module_base):  # type: ignore[misc]
        def __init__(self) -> None:
            super().__init__()
            channels = [config.base_channels * 2**level for level in range(config.depth + 1)]
            self.encoders = nn.ModuleList()
            previous = config.in_channels
            for channel_count in channels:
                self.encoders.append(DoubleConv(previous, channel_count))
                previous = channel_count
            self.pool = nn.MaxPool3d(2)
            self.upconvs = nn.ModuleList()
            self.decoders = nn.ModuleList()
            for level in range(config.depth - 1, -1, -1):
                self.upconvs.append(
                    nn.ConvTranspose3d(channels[level + 1], channels[level], 2, stride=2)
                )
                self.decoders.append(DoubleConv(channels[level] * 2, channels[level]))
            self.output = nn.Conv3d(channels[0], config.out_channels, 1)

        def forward(self, inputs: Any) -> Any:
            skips = []
            value = inputs
            for index, encoder in enumerate(self.encoders):
                value = encoder(value)
                if index < len(self.encoders) - 1:
                    skips.append(value)
                    value = self.pool(value)
            for upconv, decoder, skip in zip(
                self.upconvs, self.decoders, reversed(skips), strict=True
            ):
                value = upconv(value)
                value = decoder(torch.cat((skip, value), dim=1))
            return self.output(value)

    return UNet3D()


def _dice_loss(logits: Any, targets: Any) -> Any:
    torch = _torch()
    probabilities = torch.sigmoid(logits)
    dimensions = tuple(range(1, probabilities.ndim))
    intersection = (probabilities * targets).sum(dim=dimensions)
    denominator = probabilities.sum(dim=dimensions) + targets.sum(dim=dimensions)
    return 1.0 - ((2.0 * intersection + 1.0) / (denominator + 1.0)).mean()


def _subset_indices(subset: Any) -> list[int]:
    indices = getattr(subset, "indices", None)
    if indices is None:
        return list(range(len(subset)))
    return [int(value) for value in indices]


def _checkpoint_paths(output_path: Path) -> tuple[Path, Path]:
    return (
        output_path.parent / f"{output_path.stem}_last.pt",
        output_path.parent / f"{output_path.stem}_best.pt",
    )


def _save_training_checkpoint(
    path: Path,
    *,
    torch: Any,
    model: Any,
    optimizer: Any,
    scaler: Any | None,
    epoch: int,
    best_validation: float,
    best_state: dict[str, Any] | None,
    history: list[dict[str, float | int]],
    config: DetectorTrainingConfig,
    training_indices: list[int],
    validation_indices: list[int],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state": {name: value.detach().cpu() for name, value in model.state_dict().items()},
        "optimizer_state": optimizer.state_dict(),
        "scaler_state": scaler.state_dict() if scaler is not None else None,
        "epoch": epoch,
        "best_validation": best_validation,
        "best_state": best_state,
        "history": history,
        "unet": asdict(config.unet),
        "seed": config.seed,
        "training_indices": training_indices,
        "validation_indices": validation_indices,
    }
    torch.save(payload, path)


def train_detector(
    competition_root: str | Path,
    output_path: str | Path,
    config: DetectorTrainingConfig,
    *,
    preprocessing: PreprocessingConfig | None = None,
) -> dict[str, Any]:
    """Train a 3D U-Net and export a sigmoid TorchScript heatmap predictor."""
    torch = _torch()
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    device = torch.device(
        config.device if config.device != "cuda" or torch.cuda.is_available() else "cpu"
    )
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    last_path, best_path = _checkpoint_paths(destination)

    dataset = CentroidPatchDataset(
        competition_root,
        patch_shape_zyx=config.patch_shape_zyx,
        sigma_um=config.sigma_um,
        preprocessing=preprocessing or PreprocessingConfig(),
        jitter_voxels_zyx=config.jitter_voxels_zyx,
        seed=config.seed,
        frame_cache_size=config.frame_cache_size,
        patch_mix=config.patch_mix,
        positive_center_radius_um=config.positive_center_radius_um,
        empty_exclusion_margin_um=config.empty_exclusion_margin_um,
        augmentation=config.augmentation,
    )
    if not len(dataset):
        raise ValueError("No labeled GEFF nodes are available for detector training")
    train_view = DatasetView(dataset, train=True)
    eval_view = DatasetView(dataset, train=False)
    generator = torch.Generator().manual_seed(config.seed)
    dataset_names = sorted({dataset_name for dataset_name, _ in dataset.samples})
    split_rng = random.Random(config.seed)
    split_rng.shuffle(dataset_names)

    resumed: dict[str, Any] | None = None
    if config.resume and last_path.is_file():
        try:
            resumed = torch.load(last_path, map_location="cpu", weights_only=False)
        except TypeError:  # pragma: no cover - older torch
            resumed = torch.load(last_path, map_location="cpu")
        print(f"Resuming detector training from {last_path} (epoch={resumed['epoch']})")

    if resumed is not None:
        training_indices = [int(value) for value in resumed["training_indices"]]
        validation_indices = [int(value) for value in resumed["validation_indices"]]
        training_data = torch.utils.data.Subset(train_view, training_indices)
        validation_data = torch.utils.data.Subset(eval_view, validation_indices)
        training_datasets = {dataset.samples[index][0] for index in training_indices}
        validation_datasets = {dataset.samples[index][0] for index in validation_indices}
    elif len(dataset_names) > 1 and config.validation_fraction > 0:
        validation_dataset_count = min(
            max(1, round(len(dataset_names) * config.validation_fraction)),
            len(dataset_names) - 1,
        )
        validation_datasets = set(dataset_names[:validation_dataset_count])
        training_datasets = set(dataset_names[validation_dataset_count:])
        training_indices = [
            index
            for index, (dataset_name, _) in enumerate(dataset.samples)
            if dataset_name in training_datasets
        ]
        validation_indices = [
            index
            for index, (dataset_name, _) in enumerate(dataset.samples)
            if dataset_name in validation_datasets
        ]
        training_data = torch.utils.data.Subset(train_view, training_indices)
        validation_data = torch.utils.data.Subset(eval_view, validation_indices)
    else:
        validation_size = round(len(dataset) * config.validation_fraction)
        if len(dataset) > 1 and config.validation_fraction > 0:
            validation_size = min(max(1, validation_size), len(dataset) - 1)
        training_size = len(dataset) - validation_size
        training_split, validation_split = torch.utils.data.random_split(
            train_view, [training_size, validation_size], generator=generator
        )
        if validation_size:
            validation_indices = list(validation_split.indices)
            training_indices = list(training_split.indices)
            validation_data = torch.utils.data.Subset(eval_view, validation_indices)
            training_data = torch.utils.data.Subset(train_view, training_indices)
        else:
            training_data = training_split
            validation_data = validation_split
            training_indices = _subset_indices(training_data)
            validation_indices = _subset_indices(validation_data)
        training_datasets = set(dataset_names)
        validation_datasets = set(dataset_names) if validation_size else set()

    training_size = len(training_data)
    validation_size = len(validation_data)
    loader_kwargs: dict[str, Any] = {
        "num_workers": config.num_workers,
        "pin_memory": device.type == "cuda",
    }
    if config.num_workers > 0:
        loader_kwargs["persistent_workers"] = True

    batch_sampler: FrameGroupedBatchSampler | None = None
    if config.frame_grouped_batches:
        batch_sampler = FrameGroupedBatchSampler(
            dataset,
            training_indices,
            batch_size=config.batch_size,
            seed=config.seed,
        )
        training_loader = torch.utils.data.DataLoader(
            training_data,
            batch_sampler=batch_sampler,
            **loader_kwargs,
        )
    else:
        training_loader = torch.utils.data.DataLoader(
            training_data,
            batch_size=config.batch_size,
            shuffle=True,
            **loader_kwargs,
        )
    validation_loader = torch.utils.data.DataLoader(
        validation_data,
        batch_size=config.batch_size,
        shuffle=False,
        **loader_kwargs,
    )

    model = build_unet3d(config.unet).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    use_amp = bool(config.use_amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if use_amp else None
    bce = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(config.positive_weight, device=device))
    history: list[dict[str, float | int]] = []
    best_validation = float("inf")
    best_state: dict[str, Any] | None = None
    start_epoch = 0
    if resumed is not None:
        model.load_state_dict(resumed["model_state"])
        optimizer.load_state_dict(resumed["optimizer_state"])
        if scaler is not None and resumed.get("scaler_state") is not None:
            scaler.load_state_dict(resumed["scaler_state"])
        history = list(resumed.get("history", []))
        best_validation = float(resumed.get("best_validation", float("inf")))
        best_state = resumed.get("best_state")
        start_epoch = int(resumed["epoch"])

    print(
        f"Detector train: samples={len(dataset)} train={training_size} "
        f"val={validation_size} batch={config.batch_size} epochs={config.epochs} "
        f"frame_grouped={config.frame_grouped_batches} amp={use_amp} device={device}"
    )

    for epoch in range(start_epoch, config.epochs):
        dataset.set_epoch(epoch)
        if batch_sampler is not None:
            batch_sampler.set_epoch(epoch)
        model.train()
        training_losses: list[float] = []
        epoch_started = time.perf_counter()
        samples_seen = 0
        for step, (images, targets) in enumerate(training_loader, start=1):
            images = images.to(device=device, dtype=torch.float32, non_blocking=True)
            targets = targets.to(device=device, dtype=torch.float32, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(images)
                loss = bce(logits, targets) + _dice_loss(logits, targets)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            training_losses.append(float(loss.detach().cpu()))
            samples_seen += int(images.shape[0])
            if step % config.log_every == 0 or step == 1:
                elapsed = max(time.perf_counter() - epoch_started, 1e-6)
                print(
                    f"epoch {epoch + 1}/{config.epochs} step {step}/{len(training_loader)} "
                    f"loss={training_losses[-1]:.4f} "
                    f"samples/s={samples_seen / elapsed:.1f} elapsed_s={elapsed:.1f}"
                )
        model.eval()
        validation_losses: list[float] = []
        with torch.inference_mode():
            for images, targets in validation_loader:
                images = images.to(device=device, dtype=torch.float32)
                targets = targets.to(device=device, dtype=torch.float32)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    logits = model(images)
                    loss = bce(logits, targets) + _dice_loss(logits, targets)
                validation_losses.append(float(loss.detach().cpu()))
        train_loss = float(np.mean(training_losses)) if training_losses else float("nan")
        validation_loss = float(np.mean(validation_losses)) if validation_losses else train_loss
        history.append(
            {"epoch": epoch + 1, "train_loss": train_loss, "validation_loss": validation_loss}
        )
        if validation_loss < best_validation:
            best_validation = validation_loss
            best_state = {name: value.detach().cpu() for name, value in model.state_dict().items()}
            _save_training_checkpoint(
                best_path,
                torch=torch,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                epoch=epoch + 1,
                best_validation=best_validation,
                best_state=best_state,
                history=history,
                config=config,
                training_indices=training_indices,
                validation_indices=validation_indices,
            )
            print(f"epoch {epoch + 1}: new best validation_loss={validation_loss:.6f} -> {best_path}")
        _save_training_checkpoint(
            last_path,
            torch=torch,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            epoch=epoch + 1,
            best_validation=best_validation,
            best_state=best_state,
            history=history,
            config=config,
            training_indices=training_indices,
            validation_indices=validation_indices,
        )
        elapsed = time.perf_counter() - epoch_started
        print(
            f"epoch {epoch + 1}/{config.epochs} done train_loss={train_loss:.6f} "
            f"validation_loss={validation_loss:.6f} elapsed_s={elapsed:.1f} saved={last_path}"
        )

    if best_state is None:
        raise RuntimeError("Detector training completed without a checkpoint")
    model.load_state_dict(best_state)
    model = model.to("cpu").eval()
    checkpoint = destination.with_suffix(".pt")
    torch.save({"model_state": best_state, "unet": asdict(config.unet)}, checkpoint)

    module_base: Any = torch.nn.Module

    class SigmoidPredictor(module_base):  # type: ignore[misc]
        def __init__(self, inner: Any) -> None:
            super().__init__()
            self.inner = inner

        def forward(self, inputs: Any) -> Any:
            return torch.sigmoid(self.inner(inputs))

    example = torch.zeros((1, 1, *config.patch_shape_zyx), dtype=torch.float32)
    exported = torch.jit.trace(SigmoidPredictor(model), example)
    exported.save(str(destination))
    manifest: dict[str, Any] = {
        "artifact": str(destination),
        "checkpoint": str(checkpoint),
        "last_checkpoint": str(last_path),
        "best_checkpoint": str(best_path),
        "samples": len(dataset),
        "training_samples": training_size,
        "validation_samples": validation_size,
        "training_datasets": sorted(training_datasets),
        "validation_datasets": sorted(validation_datasets),
        "best_validation_loss": best_validation,
        "history": history,
        "config": asdict(config),
    }
    destination.with_suffix(".json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
