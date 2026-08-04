from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from biohub_tracker.models import FileTableInspection, LineageGraph, LineageNode

TABLE_SUFFIXES = {".csv", ".tsv", ".parquet", ".json", ".jsonl"}
STORE_SUFFIXES = {".zarr", ".geff"}


def _is_inside_store(path: Path) -> bool:
    return any(parent.suffix.lower() in STORE_SUFFIXES for parent in path.parents)


def discover_geff_stores(root: str | Path) -> list[Path]:
    competition_root = Path(root)
    train_root = competition_root / "train"
    if not train_root.is_dir():
        return []
    return sorted(path for path in train_root.rglob("*.geff") if path.is_dir())


def discover_annotation_files(root: str | Path) -> list[Path]:
    competition_root = Path(root)
    candidates: list[Path] = []
    candidates.extend(discover_geff_stores(competition_root))
    for path in competition_root.rglob("*") if competition_root.exists() else []:
        if not path.is_file() or path.name.lower() == "sample_submission.csv":
            continue
        if _is_inside_store(path):
            continue
        lowered = "/".join(part.lower() for part in path.parts)
        if path.suffix.lower() in TABLE_SUFFIXES and any(
            marker in lowered for marker in ("annot", "track", "label", "train")
        ):
            candidates.append(path)
    return sorted(candidates)


def inspect_geff(path: str | Path, sample_rows: int = 5) -> FileTableInspection:
    try:
        import zarr
    except ImportError as exc:  # pragma: no cover - environment error
        raise RuntimeError("GEFF support requires the 'zarr' package") from exc

    source = Path(path)
    root: Any = zarr.open_group(str(source), mode="r")
    attrs = dict(root.attrs) if hasattr(root.attrs, "keys") else {}
    geff = attrs.get("geff")
    if not isinstance(geff, dict):
        raise ValueError(f"GEFF store {source} is missing attributes.geff metadata")

    nodes: Any = root["nodes"]
    edges: Any = root["edges"]
    props: Any = nodes["props"]
    node_ids = np.asarray(nodes["ids"])
    values = {name: np.asarray(props[name]["values"]) for name in ("t", "z", "y", "x")}
    edge_ids = np.asarray(edges["ids"])
    if edge_ids.ndim != 2 or edge_ids.shape[1] != 2:
        raise ValueError(f"GEFF edges/ids must have shape (N, 2); got {edge_ids.shape}")

    n_nodes = int(node_ids.shape[0])
    for name, array in values.items():
        if array.shape[0] != n_nodes:
            raise ValueError(
                f"GEFF node prop {name!r} length {array.shape[0]} != node count {n_nodes}"
            )

    sample: list[dict[str, Any]] = []
    for index in range(min(sample_rows, n_nodes)):
        sample.append(
            {
                "node_id": int(node_ids[index]),
                "t": int(values["t"][index]),
                "z": int(values["z"][index]),
                "y": int(values["y"][index]),
                "x": int(values["x"][index]),
            }
        )
    for index in range(min(sample_rows, int(edge_ids.shape[0]))):
        sample.append(
            {
                "source_id": int(edge_ids[index, 0]),
                "target_id": int(edge_ids[index, 1]),
            }
        )

    estimated = None
    extra = geff.get("extra")
    if isinstance(extra, dict) and "estimated_number_of_nodes" in extra:
        estimated = int(extra["estimated_number_of_nodes"])

    return FileTableInspection(
        path=str(source),
        format="geff",
        columns=("node_id", "t", "z", "y", "x", "source_id", "target_id"),
        dtypes={
            "node_id": str(node_ids.dtype),
            "t": str(values["t"].dtype),
            "z": str(values["z"].dtype),
            "y": str(values["y"].dtype),
            "x": str(values["x"].dtype),
            "source_id": str(edge_ids.dtype),
            "target_id": str(edge_ids.dtype),
            "geff_version": str(geff.get("geff_version")),
            "directed": str(geff.get("directed")),
            "estimated_number_of_nodes": str(estimated),
            "labeled_nodes": str(n_nodes),
            "labeled_edges": str(int(edge_ids.shape[0])),
        },
        row_count=n_nodes + int(edge_ids.shape[0]),
        sample_rows=tuple(sample),
    )


def read_geff_graph(path: str | Path) -> LineageGraph:
    """Read the labeled nodes and directed edges from one GEFF lineage store."""
    try:
        import zarr
    except ImportError as exc:  # pragma: no cover - environment error
        raise RuntimeError("GEFF support requires the 'zarr' package") from exc

    source = Path(path)
    root: Any = zarr.open_group(str(source), mode="r")
    attrs = dict(root.attrs) if hasattr(root.attrs, "keys") else {}
    geff = attrs.get("geff")
    if not isinstance(geff, dict):
        raise ValueError(f"GEFF store {source} is missing attributes.geff metadata")
    if geff.get("directed") is False:
        raise ValueError(f"GEFF lineage graph must be directed: {source}")

    node_group: Any = root["nodes"]
    props: Any = node_group["props"]
    ids = np.asarray(node_group["ids"])
    values = {name: np.asarray(props[name]["values"]) for name in ("t", "z", "y", "x")}
    if any(array.shape != ids.shape for array in values.values()):
        raise ValueError(f"GEFF node properties do not align with node IDs in {source}")
    edge_ids = np.asarray(root["edges"]["ids"])
    if edge_ids.ndim != 2 or edge_ids.shape[1] != 2:
        raise ValueError(f"GEFF edges/ids must have shape (N, 2); got {edge_ids.shape}")

    nodes = tuple(
        LineageNode(
            node_id=int(ids[index]),
            t=int(values["t"][index]),
            z=float(values["z"][index]),
            y=float(values["y"][index]),
            x=float(values["x"][index]),
        )
        for index in range(len(ids))
    )
    node_ids = {node.node_id for node in nodes}
    edges = frozenset((int(row[0]), int(row[1])) for row in edge_ids)
    dangling = {node_id for edge in edges for node_id in edge if node_id not in node_ids}
    if dangling:
        raise ValueError(f"GEFF edges reference missing node IDs: {sorted(dangling)[:10]}")
    return LineageGraph(nodes=nodes, edges=edges)


def inspect_table(path: str | Path, sample_rows: int = 5) -> FileTableInspection:
    source = Path(path)
    if source.suffix.lower() == ".geff":
        return inspect_geff(source, sample_rows=sample_rows)
    suffix = source.suffix.lower()
    if suffix == ".csv":
        table = pd.read_csv(source)
    elif suffix == ".tsv":
        table = pd.read_csv(source, sep="\t")
    elif suffix == ".parquet":
        table = pd.read_parquet(source)
    elif suffix in {".json", ".jsonl"}:
        table = pd.read_json(source, lines=suffix == ".jsonl")
    else:
        raise ValueError(f"Unsupported table format: {source}")
    rows: list[dict[str, Any]] = json.loads(
        table.head(sample_rows).to_json(orient="records", date_format="iso")
    )
    return FileTableInspection(
        path=str(source),
        format=suffix.removeprefix("."),
        columns=tuple(str(column) for column in table.columns),
        dtypes={str(column): str(dtype) for column, dtype in table.dtypes.items()},
        row_count=len(table),
        sample_rows=tuple(rows),
    )
