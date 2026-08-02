from __future__ import annotations

from collections import Counter, defaultdict, deque
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.api.types import is_integer_dtype

from biohub_tracker.inspection import discover_competition_layout
from biohub_tracker.models import DatasetMetadata, PredictedNode, PredictionGraph
from biohub_tracker.submission.writer import NUMERIC_COLUMNS, SUBMISSION_COLUMNS
from biohub_tracker.zarr_reader import VolumeDatasetReader


class ValidationError(ValueError):
    pass


def _assert_acyclic(dataset: str, node_ids: set[int], edges: Sequence[tuple[int, int]]) -> None:
    outgoing: dict[int, list[int]] = defaultdict(list)
    indegree = {node_id: 0 for node_id in node_ids}
    for source, target in edges:
        outgoing[source].append(target)
        indegree[target] += 1
    queue = deque(node for node, degree in indegree.items() if degree == 0)
    visited = 0
    while queue:
        source = queue.popleft()
        visited += 1
        for target in outgoing[source]:
            indegree[target] -= 1
            if indegree[target] == 0:
                queue.append(target)
    if visited != len(node_ids):
        raise ValidationError(f"Dataset {dataset!r} contains a directed cycle")


def validate_graph(
    graph: PredictionGraph,
    metadata_by_dataset: Mapping[str, DatasetMetadata] | None = None,
) -> None:
    errors: list[str] = []
    node_lookup: dict[tuple[str, int], PredictedNode] = {}
    for node in graph.nodes:
        key = (node.dataset, node.node_id)
        if key in node_lookup:
            errors.append(f"duplicate node ID {node.node_id} in dataset {node.dataset!r}")
        node_lookup[key] = node
        if node.node_id < 0:
            errors.append(f"node ID must be non-negative: {key}")
        if node.t < 0 or min(node.voxel_position_zyx) < 0:
            errors.append(f"node has negative time or coordinate: {node}")
        if metadata_by_dataset is not None:
            metadata = metadata_by_dataset.get(node.dataset)
            if metadata is None:
                errors.append(f"no metadata supplied for dataset {node.dataset!r}")
            else:
                if node.t >= metadata.time_points:
                    errors.append(
                        f"node {key} time {node.t} exceeds {metadata.time_points} time points"
                    )
                for value, size, axis in zip(
                    node.voxel_position_zyx,
                    metadata.spatial_shape_zyx,
                    ("z", "y", "x"),
                    strict=True,
                ):
                    if value >= size:
                        errors.append(f"node {key} {axis}={value} outside [0, {size})")
    edge_keys: set[tuple[str, int, int]] = set()
    incoming: Counter[tuple[str, int]] = Counter()
    per_dataset_edges: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for edge in graph.edges:
        edge_key = (edge.dataset, edge.source_id, edge.target_id)
        if edge_key in edge_keys:
            errors.append(f"duplicate edge {edge_key}")
        edge_keys.add(edge_key)
        source = node_lookup.get((edge.dataset, edge.source_id))
        target = node_lookup.get((edge.dataset, edge.target_id))
        if source is None:
            errors.append(f"edge {edge_key} references missing source")
        if target is None:
            errors.append(f"edge {edge_key} references missing target")
        if edge.source_id == edge.target_id:
            errors.append(f"edge {edge_key} is a self-edge")
        if source is not None and target is not None and source.t >= target.t:
            errors.append(f"edge {edge_key} is not forward in time ({source.t} -> {target.t})")
        incoming[(edge.dataset, edge.target_id)] += 1
        per_dataset_edges[edge.dataset].append((edge.source_id, edge.target_id))
    for key, count in incoming.items():
        if count > 1:
            errors.append(f"node {key} has {count} incoming parent edges")
    if errors:
        raise ValidationError("Invalid prediction graph:\n- " + "\n- ".join(errors))
    datasets = {node.dataset for node in graph.nodes}
    for dataset in datasets:
        ids = {node.node_id for node in graph.nodes if node.dataset == dataset}
        _assert_acyclic(dataset, ids, per_dataset_edges[dataset])


def _schema_from_sample(path: str | Path) -> tuple[list[str], dict[str, str]]:
    sample = pd.read_csv(path)
    return [str(column) for column in sample.columns], {
        str(column): str(dtype) for column, dtype in sample.dtypes.items()
    }


def _validated_int(value: object, field: str) -> int:
    if not isinstance(value, (int, np.integer)):
        raise ValidationError(f"{field} must contain integer values; got {value!r}")
    return int(value)


def validate_submission(
    submission: pd.DataFrame,
    competition_root: str | Path | None = None,
    *,
    expected_dataset_names: set[str] | None = None,
    metadata_by_dataset: Mapping[str, DatasetMetadata] | None = None,
    sample_submission_path: str | Path | None = None,
) -> None:
    errors: list[str] = []
    if list(submission.columns) != SUBMISSION_COLUMNS:
        errors.append(
            f"columns must be exactly {SUBMISSION_COLUMNS}; got {list(submission.columns)}"
        )
        raise ValidationError("Invalid submission:\n- " + "\n- ".join(errors))
    if submission.isna().any().any():
        missing = submission.columns[submission.isna().any()].tolist()
        errors.append(f"missing values found in columns {missing}")
    for column in NUMERIC_COLUMNS:
        if submission[column].dtype != np.dtype(np.int64):
            errors.append(
                f"column {column!r} must have dtype int64, got {submission[column].dtype}"
            )
    if not np.array_equal(submission["id"].to_numpy(), np.arange(len(submission), dtype=np.int64)):
        errors.append("id values must be consecutive integers starting at zero")
    row_types = set(submission["row_type"].astype(str).unique())
    if not row_types <= {"node", "edge"}:
        errors.append(f"row_type contains invalid values: {sorted(row_types - {'node', 'edge'})}")
    nodes = submission[submission["row_type"] == "node"]
    edges = submission[submission["row_type"] == "edge"]
    if not nodes.empty and not (nodes[["source_id", "target_id"]] == -1).all().all():
        errors.append("every node row must use -1 for source_id and target_id")
    node_sentinels = ["node_id", "t", "z", "y", "x"]
    if not edges.empty and not (edges[node_sentinels] == -1).all().all():
        errors.append("every edge row must use -1 for node_id, t, z, y, and x")
    if competition_root is not None:
        layout = discover_competition_layout(competition_root)
        if not layout.test_stores:
            errors.append(f"no test .zarr datasets found under {Path(competition_root) / 'test'}")
        expected_dataset_names = {Path(path).stem for path in layout.test_stores}
        if layout.sample_submission:
            sample_submission_path = layout.sample_submission
        if metadata_by_dataset is None and layout.test_stores:
            reader = VolumeDatasetReader(competition_root)
            try:
                metadata_by_dataset = {
                    dataset: reader.metadata(dataset) for dataset in reader.dataset_names()
                }
            except Exception as exc:
                errors.append(f"could not load dataset metadata for bounds validation: {exc}")
    actual_datasets = set(submission["dataset"].astype(str).unique())
    if expected_dataset_names is not None and actual_datasets != expected_dataset_names:
        errors.append(
            "dataset names must exactly match discovered test datasets; "
            f"missing={sorted(expected_dataset_names - actual_datasets)}, "
            f"unexpected={sorted(actual_datasets - expected_dataset_names)}"
        )
    if submission["dataset"].astype(str).str.endswith(".zarr").any():
        errors.append("dataset names must omit the .zarr suffix")
    duplicate_nodes = nodes.duplicated(["dataset", "node_id"], keep=False)
    if duplicate_nodes.any():
        values = (
            nodes.loc[duplicate_nodes, ["dataset", "node_id"]].drop_duplicates().to_dict("records")
        )
        errors.append(f"duplicate node IDs: {values}")
    duplicate_edges = edges.duplicated(["dataset", "source_id", "target_id"], keep=False)
    if duplicate_edges.any():
        values = (
            edges.loc[duplicate_edges, ["dataset", "source_id", "target_id"]]
            .drop_duplicates()
            .to_dict("records")
        )
        errors.append(f"duplicate edges: {values}")
    graph = PredictionGraph(
        nodes=[
            PredictedNode(
                dataset=str(row.dataset),
                node_id=_validated_int(row.node_id, "node_id"),
                t=_validated_int(row.t, "t"),
                z=_validated_int(row.z, "z"),
                y=_validated_int(row.y, "y"),
                x=_validated_int(row.x, "x"),
            )
            for row in nodes.itertuples(index=False)
        ],
        edges=[],
    )
    from biohub_tracker.models import PredictedEdge

    graph.edges = [
        PredictedEdge(
            dataset=str(row.dataset),
            source_id=_validated_int(row.source_id, "source_id"),
            target_id=_validated_int(row.target_id, "target_id"),
        )
        for row in edges.itertuples(index=False)
    ]
    try:
        validate_graph(graph, metadata_by_dataset)
    except ValidationError as exc:
        errors.append(str(exc))
    if sample_submission_path is not None:
        try:
            sample_columns, sample_dtypes = _schema_from_sample(sample_submission_path)
            if sample_columns != SUBMISSION_COLUMNS:
                errors.append(
                    f"local sample schema {sample_columns} conflicts with expected "
                    f"{SUBMISSION_COLUMNS}"
                )
            generated_dtypes = {column: str(dtype) for column, dtype in submission.dtypes.items()}
            incompatible = {
                column: (sample_dtypes[column], generated_dtypes[column])
                for column in SUBMISSION_COLUMNS
                if column in sample_dtypes
                and (
                    (column in NUMERIC_COLUMNS and not is_integer_dtype(submission[column].dtype))
                    or (
                        column not in NUMERIC_COLUMNS
                        and generated_dtypes[column] != sample_dtypes[column]
                    )
                )
            }
            if incompatible:
                errors.append(f"generated dtypes conflict with local sample: {incompatible}")
        except Exception as exc:
            errors.append(f"could not compare local sample_submission.csv schema: {exc}")
    if errors:
        raise ValidationError("Invalid submission:\n- " + "\n- ".join(errors))
