from __future__ import annotations

import argparse
from pathlib import Path

from biohub_tracker.fixtures import generate_tiny_competition

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path, nargs="?", default=Path("data/sample"))
    args = parser.parse_args()
    generate_tiny_competition(args.root)
