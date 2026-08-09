#!/usr/bin/env python3
"""Build the fixed-8 association-density diagnostic from saved GEFFs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd

from biohub_pipeline.association_density import (
    FIXED8_DATASETS,
    add_bins,
    build_summaries,
    classify_hypothesis,
    diagnose_dataset,
    read_geff_tables,
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--predictions-dir", required=True, type=Path)
    result.add_argument("--ground-truth-dir", required=True, type=Path)
    result.add_argument("--output-dir", type=Path, default=Path("outputs/analysis"))
    result.add_argument(
        "--datasets",
        nargs="+",
        default=list(FIXED8_DATASETS),
        help="Dataset IDs; defaults to the repository fixed-eight set",
    )
    result.add_argument("--density-radius-um", type=float, default=15.0)
    result.add_argument("--max-match-um", type=float, default=7.0)
    result.add_argument("--scale-zyx", type=float, nargs=3, default=(1.625, 0.40625, 0.40625))
    result.add_argument(
        "--artifact-label",
        default="two-seed alpha=0.5 det=0.96875 saved selected-edge GEFF",
    )
    return result


def _fmt(value: object) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return "unavailable" if pd.isna(number) else f"{number:.4f}"


def _markdown_table(frame: pd.DataFrame) -> str:
    """Render a compact Markdown table without pandas' optional tabulate extra."""
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


def _json_safe(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def main() -> None:
    args = parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    node_metadata = []
    for dataset in args.datasets:
        pred_path = args.predictions_dir / f"{dataset}.geff"
        gt_path = args.ground_truth_dir / f"{dataset}.geff"
        if not pred_path.exists() or not gt_path.exists():
            raise FileNotFoundError(f"missing fixed-8 pair: {pred_path} / {gt_path}")
        frame, metadata = diagnose_dataset(
            dataset,
            read_geff_tables(pred_path),
            read_geff_tables(gt_path),
            scale=tuple(args.scale_zyx),
            max_match_um=args.max_match_um,
            density_radius_um=args.density_radius_um,
        )
        frames.append(frame)
        node_metadata.append(metadata)

    diagnostic, thresholds = add_bins(pd.concat(frames, ignore_index=True))
    summary, by_dataset = build_summaries(diagnostic, node_metadata)
    conclusion = classify_hypothesis(diagnostic)
    missing_fixed8 = [dataset for dataset in FIXED8_DATASETS if dataset not in args.datasets]
    diagnostic_path = args.output_dir / "association_density_diagnostic.csv"
    summary_path = args.output_dir / "association_density_summary.csv"
    dataset_path = args.output_dir / "association_density_by_dataset.csv"
    report_path = args.output_dir / "association_density_report.md"
    diagnostic.to_csv(diagnostic_path, index=False)
    summary.to_csv(summary_path, index=False)
    by_dataset.to_csv(dataset_path, index=False)

    wrong = diagnostic[~diagnostic["correct"]].copy()
    wrong["error_priority"] = (
        wrong["outcome"]
        .map(
            {
                "wrong_association": 0,
                "missing_edge": 1,
                "target_node_missed": 2,
                "source_node_missed": 3,
            }
        )
        .fillna(4)
    )
    wrong = wrong.sort_values(
        ["error_priority", "selected_edge_score", "local_density_pred"],
        ascending=[True, False, False],
        na_position="last",
    ).head(20)
    top_columns = [
        "dataset",
        "t",
        "gt_source_id",
        "gt_target_id",
        "outcome",
        "predicted_target_gt_id",
        "selected_edge_score",
        "candidate_margin",
        "selected_edge_distance_um",
        "local_density_pred",
        "density_bin",
    ]
    missing_label = ", ".join(missing_fixed8) if missing_fixed8 else "none"
    overall_node_recall = by_dataset["matched_gt_nodes"].sum() / by_dataset["num_gt_nodes"].sum()
    density_family_columns = [
        "group_type",
        "group",
        "associations",
        "errors",
        "error_rate",
        "edge_tp",
        "edge_fp",
        "edge_fn",
        "edge_precision",
        "edge_recall",
        "edge_jaccard",
    ]
    density_family_table = _markdown_table(
        summary[summary["group_type"].isin(["dataset_family", "density_bin"])][
            density_family_columns
        ]
    )
    margin_table = _markdown_table(
        summary[summary["group_type"] == "margin_bin"][
            [
                "group",
                "associations",
                "errors",
                "error_rate",
                "margin_available",
                "median_candidate_margin",
            ]
        ]
    )
    dataset_table = _markdown_table(
        by_dataset[
            [
                "dataset",
                "associations",
                "errors",
                "error_rate",
                "edge_tp",
                "edge_fp",
                "edge_fn",
                "node_recall",
            ]
        ]
    )
    report = f"""# Association density diagnostic

## Scope and provenance

- Artifact: {args.artifact_label}
- Evaluated set: {len(args.datasets)} datasets ({", ".join(args.datasets)})
- Missing from the requested fixed-eight prediction artifact: {missing_label}
- Physical scale (z, y, x): {tuple(args.scale_zyx)} micrometers per voxel
- Node matching gate: {args.max_match_um} micrometers
- Local density radius: {args.density_radius_um} micrometers around the GT target
- Density population: selected predicted nodes in the target's next frame
- Ordinary association: GT sources with exactly one outgoing edge; GT divisions are excluded
- Density quartiles: q25={thresholds[0]:.3f}, q50={thresholds[1]:.3f}, q75={thresholds[2]:.3f}

## Predeclared decision rule

`A VERIFIED` requires all of: high-density error rate at least 1.5x low density;
small-margin error rate at least 1.5x large margin; high-density/small-margin cases
contain at least 40% of association errors; and association errors outnumber errors
with a missing GT source/target detection. `B PARTIALLY VERIFIED` requires the density
effect and association-error dominance when complete candidate margins are unavailable
or another full-verification condition fails. Otherwise the result is `C REJECTED`.

## Result: {conclusion["conclusion"]}

- High- versus low-density error ratio: {_fmt(conclusion["density_error_ratio_high_vs_low"])}
- High-density error rate: {_fmt(conclusion["high_density_error_rate"])}
- Low-density error rate: {_fmt(conclusion["low_density_error_rate"])}
- Small- versus large-margin error ratio: {_fmt(conclusion["margin_error_ratio_small_vs_large"])}
- Dense + small-margin error share: {_fmt(conclusion["high_density_small_margin_error_share"])}
- Association errors (wrong/missing edge): {conclusion["association_errors"]}
- Detection-attributed errors (source/target node unmatched): {conclusion["detection_errors"]}
- Overall node recall: {_fmt(overall_node_recall)}

The saved public graph is the learned/ILP association graph before the repository's
final motion relinking, gap repair, safe-division addition, short-track filtering,
and line-fit smoothing. Therefore these numbers diagnose the learned ordinary
association stage, not an exact rescore of the final generalization-safe submission.
No detector inference, training, submission generation, or GPU job was run.

## Density and dataset-family comparisons

{density_family_table}

## Margin comparison

{margin_table}

## Candidate-margin limitation

Candidate margins available: **{str(conclusion["margin_available"]).lower()}**. A true
top-one/top-two margin requires rejected candidate edges and their `edge_prob` values.
The saved GEFFs used here retain only optimizer-selected edges, so `candidate_count`,
`best_candidate_score`, `second_candidate_score`, and `candidate_margin` are null.
`selected_edge_score` remains available but is not presented as a margin proxy.

Minimal future export: before ILP solution filtering, persist `(source_id, target_id,
edge_prob, solution)` for every next-frame candidate edge, keyed to the final GEFF node
IDs. Re-running this CPU-only script will then populate all margin bins.

This is the highest-value next experiment: export candidate alternatives during one
already-planned inference/CV run, then rerun this diagnostic on the exact eight final
prediction graphs. Do not build a new tracker until the small-margin interaction is
measured and the final postprocessed fixed-eight graphs are available.

## By dataset

{dataset_table}

## Top 20 highest-confidence wrong ordinary associations

{_markdown_table(wrong[top_columns])}

## Reproduction

```powershell
$env:PYTHONPATH='src'
.\\.venv\\Scripts\\python.exe scripts\\run_association_density_diagnostic.py `
  --predictions-dir "{args.predictions_dir}" `
  --ground-truth-dir "{args.ground_truth_dir}" `
  --output-dir "{args.output_dir}" `
  --datasets {" ".join(args.datasets)} `
  --artifact-label "{args.artifact_label}"
```

Machine-readable conclusion:

```json
{json.dumps(_json_safe(conclusion), indent=2, sort_keys=True)}
```
"""
    report_path.write_text(report, encoding="utf-8")
    print(
        json.dumps(
            _json_safe(
                {
                    "conclusion": conclusion,
                    "outputs": [
                        str(diagnostic_path),
                        str(summary_path),
                        str(dataset_path),
                        str(report_path),
                    ],
                }
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
