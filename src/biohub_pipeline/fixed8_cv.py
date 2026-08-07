"""Reproducible fixed-8 local validation for the packaged clean V106 pipeline."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from biohub_pipeline.artifacts import require_full_inputs, validation_report
from biohub_pipeline.config import PipelineConfig, load_config
from biohub_pipeline.evaluation import (
    official_spec_evaluate,
    official_spec_per_sample,
    official_spec_summarise,
)
from biohub_pipeline.inference import (
    apply_spatial_d4_patch,
    build_predict_command,
    run_prediction,
)
from biohub_pipeline.submission import write_submission_from_geff

FIXED8_DATASETS = [
    "44b6_0113de3b",
    "44b6_0b24845f",
    "44b6_341df25f",
    "44b6_e57ff5c6",
    "6bba_05b6850b",
    "6bba_05db0fb1",
    "6bba_969618f6",
    "6bba_fc83837d",
]

CURRENT_V106_REFERENCE_SCORE = 0.87892959136423
MATERIAL_DELTA = 0.005

PER_DATASET_COLUMNS = [
    "dataset",
    "edge_tp",
    "edge_fp",
    "edge_fn",
    "division_tp",
    "division_fp",
    "division_fn",
    "edge_jaccard",
    "adj_edge_jaccard",
    "division_jaccard",
    "node_recall",
    "num_pred_nodes",
    "num_gt_nodes",
    "matched_gt_nodes",
    "estimated_total_nodes",
    "total_node_ratio",
    "edge_weight",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run clean V106 inference and official-spec-lite evaluation on fixed-8"
    )
    parser.add_argument(
        "--data-dir", required=True, type=Path, help="Competition train directory"
    )
    parser.add_argument(
        "--weights-dir", required=True, type=Path, help="Root containing the weights/ tree"
    )
    parser.add_argument(
        "--support-dir", required=True, type=Path, help="Support artifact containing repo/"
    )
    parser.add_argument("--config", type=Path, default=Path("configs/clean_v106.yaml"))
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser


def validate_fixed8_inputs(data_dir: Path) -> list[tuple[str, Path, Path]]:
    """Return the exact fixed-8 inputs or fail without selecting a fallback dataset."""
    if not data_dir.is_dir():
        raise FileNotFoundError(f"fixed-8 train directory is missing: {data_dir}")

    selected = [
        (dataset, data_dir / f"{dataset}.zarr", data_dir / f"{dataset}.geff")
        for dataset in FIXED8_DATASETS
    ]
    missing_zarr = [str(zarr) for _, zarr, _ in selected if not zarr.exists()]
    missing_geff = [str(geff) for _, _, geff in selected if not geff.exists()]
    if missing_zarr or missing_geff:
        sections: list[str] = []
        if missing_zarr:
            sections.append("missing required .zarr stores:\n- " + "\n- ".join(missing_zarr))
        if missing_geff:
            sections.append("missing required .geff ground truth:\n- " + "\n- ".join(missing_geff))
        raise FileNotFoundError("fixed-8 inputs are incomplete:\n" + "\n".join(sections))
    return selected


def find_fixed8_prediction_geffs(
    repo_dir: Path, method: str, datasets: list[str] = FIXED8_DATASETS
) -> list[Path]:
    """Resolve one raw prediction GEFF per required dataset, ignoring unrelated stale output."""
    candidates = sorted((repo_dir / "predictions").glob(f"*/{method}/split_0/*.geff"))
    by_dataset: dict[str, list[Path]] = {dataset: [] for dataset in datasets}
    for path in candidates:
        if path.stem in by_dataset:
            by_dataset[path.stem].append(path)

    missing = [dataset for dataset, paths in by_dataset.items() if not paths]
    duplicates = {dataset: paths for dataset, paths in by_dataset.items() if len(paths) > 1}
    if missing:
        raise RuntimeError("fixed-8 inference produced no graph for: " + ", ".join(missing))
    if duplicates:
        details = "; ".join(
            f"{dataset}: {[str(path) for path in paths]}"
            for dataset, paths in duplicates.items()
        )
        raise RuntimeError(f"fixed-8 inference produced duplicate graphs: {details}")
    return [by_dataset[dataset][0] for dataset in datasets]


def _read_graph_tables(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    import tracksdata as td

    loaded = td.graph.IndexedRXGraph.from_geff(path)
    graph = loaded[0] if isinstance(loaded, tuple) else loaded
    nodes = pd.DataFrame(graph.node_attrs().to_dicts())
    edges = pd.DataFrame(graph.edge_attrs().to_dicts())
    if edges.empty:
        edges = pd.DataFrame(columns=["source_id", "target_id"])
    return nodes.loc[:, ["node_id", "t", "z", "y", "x"]], edges.loc[
        :, ["source_id", "target_id"]
    ]


def _estimated_total_nodes(path: Path) -> float:
    from geff import GeffMetadata

    try:
        metadata = GeffMetadata.read(path)
        value = (metadata.extra or {}).get("estimated_number_of_nodes")
        return float(value) if value is not None else float("nan")
    except Exception as exc:
        print(f"Metadata warning {path.name}: {type(exc).__name__}: {exc}")
        return float("nan")


def evaluate_tables(
    dataset: str,
    pred_nodes: pd.DataFrame,
    pred_edges: pd.DataFrame,
    gt_nodes: pd.DataFrame,
    gt_edges: pd.DataFrame,
    estimated_total_nodes: float,
    scale: tuple[float, float, float] = (1.625, 0.40625, 0.40625),
    max_distance_um: float = 7.0,
) -> dict[str, Any]:
    """Evaluate one dataset exclusively through the existing V106 metric functions."""
    result = official_spec_evaluate(
        pred_nodes,
        pred_edges,
        gt_nodes,
        gt_edges,
        scale=scale,
        max_distance_um=max_distance_um,
    )
    scores = official_spec_per_sample(result, estimated_total_nodes)
    return {
        "dataset": dataset,
        **result.__dict__,
        "estimated_total_nodes": estimated_total_nodes,
        **scores,
    }


def aggregate_metric_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if [row["dataset"] for row in rows] != FIXED8_DATASETS:
        raise ValueError("metric rows must contain the exact fixed-8 datasets in fixed order")
    return {**official_spec_summarise(rows), "datasets": list(FIXED8_DATASETS)}


def evaluate_postprocessed_predictions(
    prediction_csv: Path, data_dir: Path, config: PipelineConfig
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    predictions = pd.read_csv(prediction_csv)
    found = sorted(predictions["dataset"].unique().tolist())
    if found != sorted(FIXED8_DATASETS):
        raise RuntimeError(f"postprocessed predictions contain datasets {found}, expected fixed-8")

    scale = tuple(float(value) for value in config.postprocessing["voxel_scale_um"])
    max_distance_um = float(config.local_cv["max_match_um"])
    rows: list[dict[str, Any]] = []
    for dataset in FIXED8_DATASETS:
        part = predictions[predictions["dataset"] == dataset]
        pred_nodes = part[part["row_type"] == "node"].loc[:, ["node_id", "t", "z", "y", "x"]]
        pred_edges = part[part["row_type"] == "edge"].loc[:, ["source_id", "target_id"]]
        gt_path = data_dir / f"{dataset}.geff"
        gt_nodes, gt_edges = _read_graph_tables(gt_path)
        rows.append(
            evaluate_tables(
                dataset,
                pred_nodes,
                pred_edges,
                gt_nodes,
                gt_edges,
                _estimated_total_nodes(gt_path),
                scale=scale,
                max_distance_um=max_distance_um,
            )
        )
    return rows, aggregate_metric_rows(rows)


def _copy_raw_predictions(geffs: list[Path], output_dir: Path) -> None:
    raw_dir = output_dir / "predictions" / "raw_geff"
    raw_dir.mkdir(parents=True)
    for source in geffs:
        destination = raw_dir / source.name
        if source.is_dir():
            shutil.copytree(source, destination)
        else:
            shutil.copy2(source, destination)


def _json_default(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot JSON-encode {type(value).__name__}")


def write_metric_outputs(
    output_dir: Path,
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
    manifest: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).loc[:, PER_DATASET_COLUMNS].to_csv(
        output_dir / "per_dataset.csv", index=False
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _git_dirty() -> bool | None:
    result = subprocess.run(
        ["git", "status", "--porcelain"], capture_output=True, text=True, check=False
    )
    return bool(result.stdout.strip()) if result.returncode == 0 else None


def _gpu_name() -> str | None:
    try:
        import torch

        return torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except ImportError:
        return None


def _config_values(config: PipelineConfig) -> dict[str, dict[str, Any]]:
    return {
        "source": config.source,
        "inference": config.inference,
        "postprocessing": config.postprocessing,
        "local_cv": config.local_cv,
    }


def run_fixed8(
    data_dir: Path,
    weights_dir: Path,
    support_dir: Path,
    config_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    data_dir = data_dir.resolve()
    weights_dir = weights_dir.resolve()
    support_dir = support_dir.resolve()
    config_path = config_path.resolve()
    output_dir = output_dir.resolve()

    config = load_config(config_path)
    validate_fixed8_inputs(data_dir)
    if output_dir.exists():
        raise FileExistsError(f"fixed-8 output directory already exists: {output_dir}")

    report = validation_report(config, data_dir, weights_dir, support_dir)
    require_full_inputs(report)
    output_dir.mkdir(parents=True)
    repo_dir = support_dir / "repo"
    checkpoint_path = weights_dir / Path(str(config.inference["weights_relative"]))
    if config.inference["spatial_d4_tta"]:
        apply_spatial_d4_patch(repo_dir, str(config.inference["prediction_script"]))

    command, split_path = build_predict_command(
        config, data_dir, repo_dir, checkpoint_path, list(FIXED8_DATASETS)
    )
    started = time.perf_counter()
    run_prediction(command, repo_dir)
    geffs = find_fixed8_prediction_geffs(repo_dir, str(config.inference["method"]))
    _copy_raw_predictions(geffs, output_dir)

    prediction_csv = output_dir / "predictions" / "postprocessed_submission.csv"
    write_submission_from_geff(geffs, config, data_dir, prediction_csv)
    rows, aggregate = evaluate_postprocessed_predictions(prediction_csv, data_dir, config)
    runtime_seconds = time.perf_counter() - started
    delta = float(aggregate["score"]) - CURRENT_V106_REFERENCE_SCORE

    summary = {
        **aggregate,
        "detection_threshold": float(config.inference["detection_threshold"]),
        "checkpoint_path": str(checkpoint_path),
        "runtime_seconds": runtime_seconds,
        "reference_score": CURRENT_V106_REFERENCE_SCORE,
        "delta_vs_reference": delta,
        "reference_difference_material": abs(delta) > MATERIAL_DELTA,
    }
    manifest = {
        "schema_version": 1,
        "experiment": "fixed8_current_v106",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "datasets": list(FIXED8_DATASETS),
        "data_dir": str(data_dir),
        "support_dir": str(support_dir),
        "config_path": str(config_path),
        "config_sha256": _sha256(config_path),
        "config": _config_values(config),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "prediction_command": command,
        "split_file": str(split_path),
        "git_commit": _git_commit(),
        "git_dirty": _git_dirty(),
        "gpu_name": _gpu_name(),
        "runtime_seconds": runtime_seconds,
        "outputs": {
            "per_dataset": "per_dataset.csv",
            "summary": "summary.json",
            "postprocessed_predictions": "predictions/postprocessed_submission.csv",
            "raw_predictions": "predictions/raw_geff",
        },
    }
    write_metric_outputs(output_dir, rows, summary, manifest)
    return summary


def _print_summary(summary: dict[str, Any]) -> None:
    print("Fixed-8 current V106")
    print(f"score:      {float(summary['score']):.12f}")
    print(f"reference:  {CURRENT_V106_REFERENCE_SCORE:.12f}")
    print(f"delta:      {float(summary['delta_vs_reference']):+.12f}")
    print(
        "edge TP / FP / FN: "
        f"{summary['edge_tp']} / {summary['edge_fp']} / {summary['edge_fn']}"
    )
    print(
        "division TP / FP / FN: "
        f"{summary['division_tp']} / {summary['division_fp']} / {summary['division_fn']}"
    )
    print(f"node recall: {float(summary['node_recall']):.12f}")
    if summary["reference_difference_material"]:
        print(
            f"WARNING: |delta| exceeds {MATERIAL_DELTA:.3f}; investigate environment/config drift."
        )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = run_fixed8(
        args.data_dir,
        args.weights_dir,
        args.support_dir,
        args.config,
        args.output_dir,
    )
    _print_summary(summary)
    print(f"results:    {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
