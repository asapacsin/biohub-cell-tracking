#!/usr/bin/env python3
"""Run one instrumented fixed-8 inference and classify the association bottleneck."""

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

from biohub_pipeline.association_density import read_geff_tables
from biohub_pipeline.candidate_bottleneck import (
    analyze_dataset,
    classify,
    summarize,
)
from biohub_pipeline.config import load_config
from biohub_pipeline.fixed8_cv import (
    FIXED8_DATASETS,
    _copy_raw_predictions,
    evaluate_postprocessed_predictions,
    find_fixed8_prediction_geffs,
    validate_fixed8_inputs,
)
from biohub_pipeline.inference import (
    apply_edge_diagnostic_patch,
    apply_spatial_d4_patch,
    build_predict_command,
    run_prediction,
)
from biohub_pipeline.submission import write_submission_from_geff

CONTROL_SCORE = 0.8847464271589631


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--support-dir", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=Path("configs/clean_v106_two_seed.yaml"))
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument("--top-k", type=int, default=16)
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help="Optional dataset stems (default: fixed-8). Use holdout-8 for independent validation.",
    )
    parser.add_argument(
        "--control-score",
        type=float,
        default=None,
        help="Optional control score override for report delta (default: fixed-8 0.96875 control).",
    )
    parser.add_argument(
        "--skip-fixed8-validation",
        action="store_true",
        help="Skip validate_fixed8_inputs when running a non-fixed-8 dataset list.",
    )
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    if result.returncode == 0:
        return result.stdout.strip()
    return "unknown"


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and (pd.isna(value) or value in (float("inf"), float("-inf"))):
        return None
    if hasattr(value, "item"):
        return _json_safe(value.item())
    return value


def _markdown(frame: pd.DataFrame) -> str:
    columns = [str(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        values = []
        for value in row:
            if pd.isna(value):
                values.append("")
            elif isinstance(value, float):
                values.append(f"{value:.6g}")
            else:
                values.append(str(value).replace("|", "\\|"))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _write_report(
    output: Path,
    score: dict[str, Any],
    decision: dict[str, Any],
    summary: pd.DataFrame,
    by_dataset: pd.DataFrame,
    runtime_seconds: float,
    control_score: float = CONTROL_SCORE,
) -> None:
    overall = summary[summary["group_type"] == "overall"].iloc[0]
    comparisons = summary[summary["group_type"].isin(["dataset_family", "density_bin"])][
        [
            "group_type",
            "group",
            "ordinary_associations",
            "errors",
            "error_rate",
            "causal_errors",
            "scorer_ranking",
            "candidate_threshold",
            "ilp_global",
            "postprocessing_removed",
            "candidate_recall",
            "target_top1_rate",
        ]
    ]
    dataset_columns = [
        column
        for column in [
            "group",
            "ordinary_associations",
            "errors",
            "error_rate",
            "scorer_ranking",
            "candidate_threshold",
            "ilp_global",
            "postprocessing_removed",
            "detected_node_recall",
            "final_node_recall",
        ]
        if column in by_dataset.columns
    ]
    report = f"""# Candidate-edge bottleneck experiment

## Final classification: {decision["classification"]}

## Hypothesis and predeclared rule

The learned edge scorer is the dominant ordinary-association failure mechanism, rather
than node detection, the 0.5 candidate threshold, ILP/global assignment, or final
postprocessing. Support required at least 20 endpoint-matched causal errors, at least
60% attributed to scorer ranking, and a lead of at least 20 percentage points over the
next mechanism. Ranking below 40%, or any competing mechanism at 40% or more, rejected it.

## Control and experiment

- Control: generalization-safe two-seed recipe C, alpha=0.5, det=0.96875, safe divisions
  ON, gap2 OFF, DeepCenter OFF; reference score `{control_score:.12f}`.
- Experiment: identical inference/postprocessing with opt-in top-16 pre-gate blended and
  per-seed score capture. Instrumentation does not alter candidate construction or ILP.
- Instrumented score: `{float(score["score"]):.12f}`; delta from control
  `{float(score["score"]) - control_score:+.12f}`.
- Runtime: `{runtime_seconds / 60:.2f}` minutes.
- GPU inference required: yes, one fixed-8 run.

## Main causal result

- Ordinary associations: {int(overall["ordinary_associations"])}
- Endpoint-matched causal errors: {int(overall["causal_errors"])}
- Scorer-ranking failures: {int(overall["scorer_ranking"])}
  ({float(overall["scorer_ranking_share"]):.2%})
- Candidate-threshold failures: {int(overall["candidate_threshold"])}
  ({float(overall["candidate_threshold_share"]):.2%})
- ILP/global failures: {int(overall["ilp_global"])}
  ({float(overall["ilp_global_share"]):.2%})
- Postprocessing removals: {int(overall["postprocessing_removed"])}
  ({float(overall["postprocessing_removed_share"]):.2%})
- Detection misses: {int(overall["detection_miss"])}
- Candidate recall with detected endpoints: {float(overall["candidate_recall"]):.2%}
- Final ordinary-edge recall with detected endpoints: {float(overall["final_edge_recall"]):.2%}

## Dense-scene and family checks

{_markdown(comparisons)}

## Per dataset

{_markdown(by_dataset[dataset_columns])}

## Engineering decision

This experiment changes no production setting. The result determines whether the next
single action should target learned edge ranking or a different stage.

**Next action:** {decision["recommended_next_action"]}

See the machine-readable `decision.json` for the predeclared classification inputs.
"""
    (output / "report.md").write_text(report, encoding="utf-8")


def _evaluate_custom(
    prediction_csv: Path, data_dir: Path, config, datasets: list[str]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Official-spec evaluation for an arbitrary ordered dataset list."""
    from biohub_pipeline.evaluation import official_spec_summarise
    from biohub_pipeline.fixed8_cv import (
        _estimated_total_nodes,
        _read_graph_tables,
        evaluate_tables,
    )

    predictions = pd.read_csv(prediction_csv)
    found = sorted(predictions["dataset"].unique().tolist())
    if found != sorted(datasets):
        raise RuntimeError(f"predictions datasets {found} != expected {sorted(datasets)}")
    scale = tuple(float(value) for value in config.postprocessing["voxel_scale_um"])
    max_distance_um = float(config.local_cv["max_match_um"])
    rows: list[dict[str, Any]] = []
    for dataset in datasets:
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
    return rows, {**official_spec_summarise(rows), "datasets": list(datasets)}


def main() -> None:
    args = _parser().parse_args()
    if args.top_k < 2:
        raise ValueError("--top-k must be at least 2")
    data_dir = args.data_dir.resolve()
    support_dir = args.support_dir.resolve()
    config_path = args.config.resolve()
    output = args.output_dir.resolve()
    work = args.work_dir.resolve()
    datasets = list(args.datasets) if args.datasets else list(FIXED8_DATASETS)
    control_score = CONTROL_SCORE if args.control_score is None else float(args.control_score)
    if not args.skip_fixed8_validation and datasets == list(FIXED8_DATASETS):
        validate_fixed8_inputs(data_dir)
    else:
        missing = [
            name
            for name in datasets
            if not (data_dir / f"{name}.zarr").exists() or not (data_dir / f"{name}.geff").exists()
        ]
        if missing:
            raise FileNotFoundError("missing dataset inputs: " + ", ".join(missing))
    if output.exists():
        raise FileExistsError(f"experiment output already exists: {output}")
    if work.exists():
        raise FileExistsError(f"experiment work directory already exists: {work}")
    output.mkdir(parents=True)
    work.mkdir(parents=True)
    shutil.copy2(config_path, output / "experiment_config.yaml")
    repo_dir = work / "support_repo"
    shutil.copytree(support_dir / "repo", repo_dir)

    config = load_config(config_path)
    if config.inference["spatial_d4_tta"]:
        apply_spatial_d4_patch(repo_dir, str(config.inference["prediction_script"]))
    primary = support_dir / Path(str(config.inference["weights_relative"]))
    command, split_path = build_predict_command(
        config, data_dir, repo_dir, primary, datasets
    )
    apply_edge_diagnostic_patch(repo_dir, str(config.inference["prediction_script"]))
    capture_dir = output / "candidate_capture"
    command.extend(
        [
            "--edge-diagnostic-dir",
            str(capture_dir),
            "--edge-diagnostic-top-k",
            str(args.top_k),
        ]
    )
    (output / "command.txt").write_text(subprocess.list2cmdline(command) + "\n", encoding="utf-8")

    started = time.perf_counter()
    run_prediction(command, repo_dir)
    geffs = find_fixed8_prediction_geffs(
        repo_dir, str(config.inference["method"]), datasets
    )
    raw_dir = _copy_raw_predictions(geffs, output, config_path=config_path)
    prediction_csv = output / "predictions" / "postprocessed_submission.csv"
    write_submission_from_geff(geffs, config, data_dir, prediction_csv)
    if datasets == list(FIXED8_DATASETS):
        metric_rows, score = evaluate_postprocessed_predictions(
            prediction_csv, data_dir, config
        )
    else:
        metric_rows, score = _evaluate_custom(prediction_csv, data_dir, config, datasets)
    runtime_seconds = time.perf_counter() - started
    pd.DataFrame(metric_rows).to_csv(output / "metric_by_dataset.csv", index=False)
    (output / "score_summary.json").write_text(
        json.dumps(_json_safe(score), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    final_predictions = pd.read_csv(prediction_csv)
    frames = []
    metadata = []
    for dataset in datasets:
        frame, dataset_metadata = analyze_dataset(
            dataset,
            capture_dir / dataset,
            raw_dir / f"{dataset}.geff",
            read_geff_tables(data_dir / f"{dataset}.geff"),
            final_predictions,
            scale=tuple(float(value) for value in config.postprocessing["voxel_scale_um"]),
            max_match_um=float(config.local_cv["max_match_um"]),
        )
        frames.append(frame)
        metadata.append(dataset_metadata)
    diagnostic = pd.concat(frames, ignore_index=True)
    summary, by_dataset = summarize(diagnostic, metadata)
    decision = classify(summary)
    diagnostic.to_csv(output / "candidate_edge_diagnostic.csv", index=False)
    summary.to_csv(output / "cause_summary.csv", index=False)
    by_dataset.to_csv(output / "cause_by_dataset.csv", index=False)
    (output / "decision.json").write_text(
        json.dumps(_json_safe(decision), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    secondary = support_dir / Path(str(config.inference["ensemble_weights_relative"]))
    metadata_doc = {
        "schema_version": 1,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "experiment": "candidate_edge_bottleneck_v1",
        "git_commit": _git_commit(),
        "datasets": list(datasets),
        "config": str(config_path),
        "config_sha256": _sha256(config_path),
        "primary_checkpoint": str(primary),
        "primary_checkpoint_sha256": _sha256(primary),
        "secondary_checkpoint": str(secondary),
        "secondary_checkpoint_sha256": _sha256(secondary),
        "instrumented_predictor_sha256": _sha256(
            repo_dir / str(config.inference["prediction_script"])
        ),
        "split_file": str(split_path),
        "top_k": args.top_k,
        "control_score": control_score,
        "experiment_score": score["score"],
        "runtime_seconds": runtime_seconds,
        "prediction_equivalent_to_control": abs(float(score["score"]) - control_score) < 1e-12,
    }
    (output / "metadata.json").write_text(
        json.dumps(_json_safe(metadata_doc), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_report(
        output,
        score,
        decision,
        summary,
        by_dataset,
        runtime_seconds,
        control_score=control_score,
    )
    print(json.dumps(_json_safe({"score": score, "decision": decision}), indent=2))


if __name__ == "__main__":
    main()
