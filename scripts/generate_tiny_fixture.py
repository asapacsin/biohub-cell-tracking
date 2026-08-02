from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from biohub_tracker.fixtures import tiny_expected_graph
from biohub_tracker.submission import build_submission, write_submission


def generate_fixture(root: Path) -> None:
    import zarr

    test_root = root / "test"
    test_root.mkdir(parents=True, exist_ok=True)
    store_path = test_root / "tiny.zarr"
    group = zarr.open_group(str(store_path), mode="w")
    image = np.zeros((3, 1, 4, 32, 32), dtype=np.uint16)
    image[0, 0, 2, 10, 10] = 100
    image[0, 0, 3, 20, 20] = 100
    image[1, 0, 2, 11, 10] = 100
    image[1, 0, 3, 21, 20] = 100
    image[2, 0, 2, 12, 9] = 100
    image[2, 0, 2, 12, 12] = 100
    image[2, 0, 3, 22, 20] = 100
    if hasattr(group, "create_array"):
        group.create_array("0", data=image, chunks=(1, 1, 2, 16, 16))
    else:  # Zarr 2
        group.create_dataset("0", data=image, chunks=(1, 1, 2, 16, 16))
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
    table = build_submission([tiny_expected_graph()])
    write_submission(table, root / "sample_submission.csv")
    annotation = table[table["row_type"] == "node"].copy()
    (root / "train").mkdir(exist_ok=True)
    annotation.to_csv(root / "train" / "tracking_annotations.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path, nargs="?", default=Path("data/sample"))
    args = parser.parse_args()
    generate_fixture(args.root)

