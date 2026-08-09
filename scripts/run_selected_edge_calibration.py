#!/usr/bin/env python3
"""Run the CPU-only selected-edge confidence calibration diagnostic."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from biohub_pipeline.selected_edge_calibration import (
    DEFAULT_BUDGET_FRACTION,
    DEFAULT_MIN_CORRECTNESS_AUROC,
    DEFAULT_MIN_ERROR_CAPTURE,
    evaluate_calibration,
    summarize_groups,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--diagnostic-csv",
        type=Path,
        default=Path("outputs/analysis/association_density_diagnostic.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/experiments/selected_edge_calibration"),
    )
    parser.add_argument("--budget-fraction", type=float, default=DEFAULT_BUDGET_FRACTION)
    parser.add_argument("--min-error-capture", type=float, default=DEFAULT_MIN_ERROR_CAPTURE)
    parser.add_argument(
        "--min-correctness-auroc",
        type=float,
        default=DEFAULT_MIN_CORRECTNESS_AUROC,
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if hasattr(value, "item"):
        return _json_safe(value.item())
    return value


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(_json_safe(value), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


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


def _write_report(output: Path, metrics: dict[str, Any], groups: pd.DataFrame) -> None:
    dataset_rows = groups[groups["group_type"] == "dataset"]
    dataset_columns = [
        "group",
        "scored_associations",
        "scored_errors",
        "error_prevalence",
        "budget_count",
        "captured_errors",
        "error_capture",
        "intervention_precision",
        "correctness_auroc",
    ]
    report = f"""# Selected-edge confidence calibration

## Final classification: {metrics["classification"]}

## Hypothesis and predeclared rule

Selected-edge confidence alone can identify a sufficiently concentrated intervention
set to justify confidence calibration as the next production lever. The fixed budget
is {float(metrics["budget_fraction"]):.1%} of scored ordinary associations. Support
requires at least {float(metrics["min_error_capture"]):.1%} error capture and correctness
AUROC of at least {float(metrics["min_correctness_auroc"]):.2f}.

## Control and experiment

- Control: the existing selected-edge scores from the four labeled public two-seed
  overlaps in `association_density_diagnostic.csv`.
- Experiment: rank scored ordinary associations from lowest to highest selected-edge
  confidence and inspect the predeclared bottom-budget set.
- Neural inference required: no; this is a CPU-only analysis of saved artifacts.
- Production inference and postprocessing were not changed.

## Result

- Scored ordinary associations: {int(metrics["scored_associations"]):,}
- Scored errors: {int(metrics["scored_errors"]):,}
- Intervention budget: {int(metrics["budget_count"]):,} associations
- Captured errors: {int(metrics["captured_errors"]):,} / {int(metrics["scored_errors"]):,}
  ({float(metrics["error_capture"]):.2%})
- Intervention precision: {float(metrics["intervention_precision"]):.2%}
- Precision lift over random: {float(metrics["precision_lift_vs_random"]):.2f}x
- Correctness AUROC: {float(metrics["correctness_auroc"]):.6f}
- Maximum selected score inside the intervention set:
  {float(metrics["confidence_threshold"]):.9f}

The low-confidence tail is enriched for errors, but it misses nearly three quarters
of scored errors and fails both predeclared practical thresholds. Confidence-only
calibration is therefore not the next production lever.

## Per dataset

{_markdown(dataset_rows[dataset_columns])}

## Decision

Do not change the production/Kaggle recipe. {metrics["recommended_next_action"]}
"""
    (output / "report.md").write_text(report, encoding="utf-8")


def main() -> None:
    args = _parser().parse_args()
    source = args.diagnostic_csv.resolve()
    output = args.output_dir.resolve()
    if not source.is_file():
        raise FileNotFoundError(f"diagnostic CSV does not exist: {source}")
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"experiment output already exists: {output}")
    output.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(source)
    metrics, selected = evaluate_calibration(
        frame,
        budget_fraction=args.budget_fraction,
        min_error_capture=args.min_error_capture,
        min_correctness_auroc=args.min_correctness_auroc,
    )
    groups = summarize_groups(
        frame,
        ("dataset_family", "density_bin", "dataset"),
        budget_fraction=args.budget_fraction,
    )
    config = {
        "schema_version": 1,
        "hypothesis": (
            "Selected-edge confidence can identify enough association errors at a "
            "predeclared intervention budget to justify confidence-only calibration."
        ),
        "budget_fraction": args.budget_fraction,
        "min_error_capture": args.min_error_capture,
        "min_correctness_auroc": args.min_correctness_auroc,
        "score_column": "selected_edge_score",
        "correct_column": "correct",
    }
    metadata = {
        "schema_version": 1,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "experiment": "selected_edge_calibration_v1",
        "git_commit": _git_commit(),
        "source": str(source),
        "source_rows": len(frame),
        "gpu_inference_required": False,
        "production_recipe_changed": False,
    }
    command = subprocess.list2cmdline(
        [
            "python",
            "scripts/run_selected_edge_calibration.py",
            "--diagnostic-csv",
            str(args.diagnostic_csv),
            "--output-dir",
            str(args.output_dir),
        ]
    )

    _write_json(output / "experiment_config.json", config)
    _write_json(output / "metadata.json", metadata)
    _write_json(output / "metrics.json", metrics)
    selected.to_csv(output / "selected_intervention_rows.csv", index=False)
    groups.to_csv(output / "metrics_by_group.csv", index=False)
    (output / "command.txt").write_text(command + "\n", encoding="utf-8")
    _write_report(output, metrics, groups)
    print(json.dumps(_json_safe(metrics), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
