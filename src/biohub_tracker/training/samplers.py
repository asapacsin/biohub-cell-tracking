"""Batch samplers that preserve Zarr frame locality during detector training."""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Iterator
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from biohub_tracker.training.data import CentroidPatchDataset


class FrameGroupedBatchSampler:
    """Yield batches whose cells all come from a single ``(dataset, t)`` frame.

    Indices yielded are positions into a ``torch.utils.data.Subset`` (not raw
    dataset indices). Frame order and within-frame cell order are reshuffled
    every epoch via :meth:`set_epoch`.
    """

    def __init__(
        self,
        dataset: CentroidPatchDataset,
        subset_indices: list[int],
        *,
        batch_size: int,
        seed: int,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.dataset = dataset
        self.batch_size = batch_size
        self.seed = seed
        self.epoch = 0
        self.groups: dict[tuple[str, int], list[int]] = defaultdict(list)
        for subset_position, dataset_index in enumerate(subset_indices):
            dataset_name, node_id = dataset.samples[dataset_index]
            t = int(dataset.nodes_by_id[dataset_name][node_id].t)
            self.groups[(dataset_name, t)].append(subset_position)

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed + self.epoch)
        frame_keys = list(self.groups)
        rng.shuffle(frame_keys)
        for key in frame_keys:
            indices = self.groups[key].copy()
            rng.shuffle(indices)
            for start in range(0, len(indices), self.batch_size):
                yield indices[start : start + self.batch_size]

    def __len__(self) -> int:
        return sum(
            (len(indices) + self.batch_size - 1) // self.batch_size
            for indices in self.groups.values()
        )
