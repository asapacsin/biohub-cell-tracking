from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class ArrayMetadata:
    path: str
    shape: tuple[int, ...]
    dtype: str
    chunks: tuple[int, ...] | None


@dataclass(frozen=True, slots=True)
class DatasetMetadata:
    name: str
    store_path: str
    array_path: str
    shape: tuple[int, ...]
    axes: tuple[str, ...]
    dtype: str
    chunks: tuple[int, ...] | None
    time_points: int
    channel_count: int
    voxel_spacing_zyx: tuple[float, float, float]
    multiscale_levels: tuple[str, ...] = ()
    arrays: tuple[ArrayMetadata, ...] = ()
    coordinate_transformations: tuple[dict[str, Any], ...] = ()
    attributes: dict[str, Any] = field(default_factory=dict, compare=False)

    @property
    def spatial_shape_zyx(self) -> tuple[int, int, int]:
        sizes = dict(zip(self.axes, self.shape, strict=True))
        try:
            return sizes["z"], sizes["y"], sizes["x"]
        except KeyError as exc:
            raise ValueError(f"Dataset {self.name!r} does not expose z, y, x axes") from exc


@dataclass(frozen=True, slots=True)
class PredictedNode:
    dataset: str
    node_id: int
    t: int
    z: int
    y: int
    x: int

    @property
    def voxel_position_zyx(self) -> tuple[int, int, int]:
        return self.z, self.y, self.x


@dataclass(frozen=True, slots=True)
class PredictedEdge:
    dataset: str
    source_id: int
    target_id: int


@dataclass(slots=True)
class DetectionCandidate:
    dataset: str
    t: int
    z: float
    y: float
    x: float
    score: float
    volume: float | None = None
    intensity: float | None = None
    appearance_embedding: tuple[float, ...] | None = None
    annotation_id: int | None = None


@dataclass(slots=True)
class PredictionGraph:
    nodes: list[PredictedNode] = field(default_factory=list)
    edges: list[PredictedEdge] = field(default_factory=list)


@dataclass(slots=True)
class NodeIdAllocator:
    next_id: int = 1

    def allocate(self) -> int:
        node_id = self.next_id
        self.next_id += 1
        return node_id


@dataclass(frozen=True, slots=True)
class CompetitionLayout:
    root: str
    sample_submission: str | None
    test_stores: tuple[str, ...]
    train_stores: tuple[str, ...]
    annotation_files: tuple[str, ...]
    other_files: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class FileTableInspection:
    path: str
    format: str
    columns: tuple[str, ...]
    dtypes: dict[str, str]
    row_count: int | None
    sample_rows: tuple[dict[str, Any], ...]


@dataclass(frozen=True, slots=True)
class LineageNode:
    node_id: int
    t: int
    z: float
    y: float
    x: float

    @property
    def position_zyx(self) -> tuple[float, float, float]:
        return self.z, self.y, self.x


@dataclass(frozen=True, slots=True)
class LineageGraph:
    nodes: tuple[LineageNode, ...]
    edges: frozenset[tuple[int, int]]
