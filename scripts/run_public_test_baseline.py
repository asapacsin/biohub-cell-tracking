#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    sys.path.insert(0, str(ROOT / "src"))
    from biohub_tracker.baseline_pipeline import load_baseline_config, run_public_test_baseline

    parser = argparse.ArgumentParser(description="Run public-test baseline detect+track.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=ROOT / "data" / "competition",
        help="Competition root containing test/*.zarr and sample_submission.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "public_test_baseline",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "baseline.yaml",
    )
    parser.add_argument("--video-id", action="append", default=None)
    parser.add_argument("--start-frame", type=int, default=None)
    parser.add_argument("--end-frame", type=int, default=None)
    parser.add_argument(
        "--save-visualizations",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    config = load_baseline_config(args.config)
    path = run_public_test_baseline(
        args.input_dir,
        args.output_dir,
        config,
        video_ids=args.video_id,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        save_visualizations=args.save_visualizations,
        overwrite=args.overwrite,
    )
    print(path)


if __name__ == "__main__":
    main()
