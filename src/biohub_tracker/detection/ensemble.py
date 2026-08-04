from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray


class HeatmapPredictor(Protocol):
    def predict(self, frame_zyx: NDArray[np.float32]) -> NDArray[np.float32]: ...


class TorchScriptHeatmapPredictor:
    """Thin optional adapter; PyTorch is required only when learned inference is selected."""

    def __init__(self, path: str | Path, *, device: str = "cpu") -> None:
        try:
            import torch  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - optional runtime
            raise RuntimeError(
                "Learned detection requires PyTorch; install the 'ml' extra or use method=blob"
            ) from exc
        self._torch: Any = torch
        self._device = device
        self._model = torch.jit.load(str(path), map_location=device).eval()

    def predict(self, frame_zyx: NDArray[np.float32]) -> NDArray[np.float32]:
        tensor = self._torch.from_numpy(frame_zyx[None, None]).to(self._device)
        with self._torch.inference_mode():
            output = self._model(tensor)
        if isinstance(output, dict):
            output = output.get("heatmap")
        array = np.asarray(output.detach().cpu(), dtype=np.float32)
        if array.ndim == 5 and array.shape[:2] == (1, 1):
            array = array[0, 0]
        if array.shape != frame_zyx.shape:
            raise ValueError(
                f"Detector returned heatmap shape {array.shape}; expected {frame_zyx.shape}"
            )
        return array


class SlidingWindowHeatmapPredictor:
    """Run a patch-trained predictor over a full frame and average overlapping heatmaps."""

    def __init__(
        self,
        predictor: HeatmapPredictor,
        *,
        window_shape_zyx: tuple[int, int, int],
        overlap: float = 0.25,
    ) -> None:
        if any(size < 1 for size in window_shape_zyx):
            raise ValueError("window_shape_zyx must contain positive sizes")
        if not 0 <= overlap < 1:
            raise ValueError("sliding-window overlap must be in [0, 1)")
        self.predictor = predictor
        self.window_shape_zyx = window_shape_zyx
        self.overlap = overlap

    @staticmethod
    def _starts(length: int, window: int, overlap: float) -> list[int]:
        if length <= window:
            return [0]
        stride = max(1, round(window * (1 - overlap)))
        starts = list(range(0, length - window + 1, stride))
        if starts[-1] != length - window:
            starts.append(length - window)
        return starts

    def predict(self, frame_zyx: NDArray[np.float32]) -> NDArray[np.float32]:
        original_shape = frame_zyx.shape
        padded_shape = tuple(
            max(size, window)
            for size, window in zip(original_shape, self.window_shape_zyx, strict=True)
        )
        padded = np.zeros(padded_shape, dtype=np.float32)
        source_slices = tuple(slice(0, size) for size in original_shape)
        padded[source_slices] = frame_zyx
        total = np.zeros(padded_shape, dtype=np.float32)
        counts = np.zeros(padded_shape, dtype=np.float32)
        starts = [
            self._starts(length, window, self.overlap)
            for length, window in zip(padded_shape, self.window_shape_zyx, strict=True)
        ]
        for z in starts[0]:
            for y in starts[1]:
                for x in starts[2]:
                    slices = (
                        slice(z, z + self.window_shape_zyx[0]),
                        slice(y, y + self.window_shape_zyx[1]),
                        slice(x, x + self.window_shape_zyx[2]),
                    )
                    predicted = self.predictor.predict(padded[slices])
                    total[slices] += predicted
                    counts[slices] += 1.0
        return (total[source_slices] / counts[source_slices]).astype(np.float32, copy=False)


def predict_ensemble_heatmap(
    frame_zyx: NDArray[np.float32],
    predictors: Sequence[HeatmapPredictor],
    *,
    tta_flips: bool = True,
) -> NDArray[np.float32]:
    """Average seed/model predictions and optional axis-flip test-time augmentation."""
    if not predictors:
        raise ValueError("At least one learned detector predictor is required")
    flips: tuple[tuple[int, ...], ...] = ((), (1,), (2,), (1, 2)) if tta_flips else ((),)
    predictions: list[NDArray[np.float32]] = []
    for predictor in predictors:
        for axes in flips:
            transformed = np.flip(frame_zyx, axis=axes).copy() if axes else frame_zyx
            predicted = np.asarray(predictor.predict(transformed), dtype=np.float32)
            if predicted.shape != frame_zyx.shape:
                raise ValueError(
                    f"Detector returned heatmap shape {predicted.shape}; expected {frame_zyx.shape}"
                )
            predictions.append(np.flip(predicted, axis=axes) if axes else predicted)
    return np.mean(np.stack(predictions), axis=0, dtype=np.float32)
