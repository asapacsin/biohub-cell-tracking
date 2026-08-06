"""Command-line entry point for the selected clean V106 pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from biohub_pipeline.artifacts import require_full_inputs, validation_report
from biohub_pipeline.config import load_config
from biohub_pipeline.inference import (
    apply_spatial_d4_patch,
    build_predict_command,
    list_stems,
    run_prediction,
)
from biohub_pipeline.submission import write_submission_from_geff


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run or validate the clean Biohub V106 pipeline")
    parser.add_argument(
        "--data-dir", type=Path, help="Directory containing competition test .zarr stores"
    )
    parser.add_argument(
        "--weights-dir", type=Path, help="Root containing the upstream weights/ tree"
    )
    parser.add_argument(
        "--support-dir", type=Path, help="Extracted support artifact containing repo/"
    )
    parser.add_argument("--config", type=Path, default=Path("configs/clean_v106.yaml"))
    parser.add_argument("--output", type=Path, default=Path("submission.csv"))
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration, paths, and dependency availability without loading the model",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    data_dir = args.data_dir.resolve() if args.data_dir is not None else None
    weights_dir = args.weights_dir.resolve() if args.weights_dir is not None else None
    support_dir = args.support_dir.resolve() if args.support_dir is not None else None
    output = args.output.resolve()
    report = validation_report(config, data_dir, weights_dir, support_dir)
    if args.dry_run:
        report["mode"] = "dry-run"
        report["model_loaded"] = False
        report["inference_started"] = False
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    require_full_inputs(report)
    assert data_dir is not None and weights_dir is not None and support_dir is not None
    repo_dir = support_dir / "repo"
    weights_path = weights_dir / Path(str(config.inference["weights_relative"]))
    stems = list_stems(data_dir)
    if config.inference["spatial_d4_tta"]:
        apply_spatial_d4_patch(repo_dir, str(config.inference["prediction_script"]))
    command, _ = build_predict_command(config, data_dir, repo_dir, weights_path, stems)
    run_prediction(command, repo_dir)
    method = str(config.inference["method"])
    geffs = sorted((repo_dir / "predictions").glob(f"*/{method}/split_0/*.geff"))
    if len(geffs) != len(stems):
        raise RuntimeError(f"expected {len(stems)} prediction graphs, found {len(geffs)}")
    summary = write_submission_from_geff(geffs, config, data_dir, output)
    print(json.dumps({"output": str(output), **summary}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
