from __future__ import annotations

from pathlib import Path

import numpy as np

from biohub_tracker.models import PredictedEdge, PredictedNode, PredictionGraph
from biohub_tracker.submission.writer import build_submission, write_submission


def tiny_expected_graph(dataset: str = "tiny") -> PredictionGraph:
    """Deterministic graph fixture; it is not produced by a detector or tracker."""
    nodes = [
        PredictedNode(dataset, 1, 0, 2, 10, 10),
        PredictedNode(dataset, 2, 0, 3, 20, 20),
        PredictedNode(dataset, 3, 1, 2, 11, 10),
        PredictedNode(dataset, 4, 1, 3, 21, 20),
        PredictedNode(dataset, 5, 2, 2, 12, 9),
        PredictedNode(dataset, 6, 2, 2, 12, 12),
        PredictedNode(dataset, 7, 2, 3, 22, 20),
    ]
    edges = [
        PredictedEdge(dataset, 1, 3),
        PredictedEdge(dataset, 2, 4),
        PredictedEdge(dataset, 3, 5),
        PredictedEdge(dataset, 3, 6),
        PredictedEdge(dataset, 4, 7),
    ]
    return PredictionGraph(nodes=nodes, edges=edges)


def _write_tiny_zarr(store_path: Path, image: np.ndarray) -> None:
    import zarr

    store_path.parent.mkdir(parents=True, exist_ok=True)
    group = zarr.open_group(str(store_path), mode="w")
    if hasattr(group, "create_array"):
        group.create_array("0", data=image, chunks=(1, 1, 2, 16, 16))
    else:  # pragma: no cover - Zarr 2 compatibility
        group.create_dataset(  # type: ignore[attr-defined]
            "0", data=image, chunks=(1, 1, 2, 16, 16)
        )
    group.attrs["multiscales"] = [
        {
            "version": "0.4",
            "axes": [
                {"name": "t", "type": "time"},
                {"name": "c", "type": "channel"},
                {"name": "z", "type": "space", "unit": "micrometer"},
                {"name": "y", "type": "space", "unit": "micrometer"},
                {"name": "x", "type": "space", "unit": "micrometer"},
            ],
            "datasets": [
                {
                    "path": "0",
                    "coordinateTransformations": [
                        {"type": "scale", "scale": [1.0, 1.0, 2.0, 0.5, 0.5]}
                    ],
                }
            ],
        }
    ]


def generate_tiny_competition(root: str | Path) -> Path:
    """Create deterministic official-shape infrastructure input, not tracker output."""
    destination = Path(root)
    image = np.zeros((3, 1, 4, 32, 32), dtype=np.uint16)
    for node in tiny_expected_graph().nodes:
        image[node.t, 0, node.z, node.y, node.x] = 100

    _write_tiny_zarr(destination / "test" / "tiny.zarr", image)
    _write_tiny_zarr(destination / "train" / "tiny_training.zarr", image)

    table = build_submission([tiny_expected_graph()])
    write_submission(table, destination / "sample_submission.csv")
    annotations = table[table["row_type"] == "node"].copy()
    annotations.to_csv(destination / "train" / "tracking_annotations.csv", index=False)
    return destination
