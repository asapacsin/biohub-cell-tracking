from __future__ import annotations

import pandas as pd

from biohub_pipeline.detection_z_shift import shift_node_z


def test_shift_node_z_only_moves_node_rows() -> None:
    frame = pd.DataFrame(
        {
            "row_type": ["node", "edge"],
            "z": [10.0, 0.0],
            "y": [1.0, 0.0],
            "x": [2.0, 0.0],
        }
    )
    out = shift_node_z(frame, 2.0)
    assert float(out.loc[0, "z"]) == 12.0
    assert float(out.loc[1, "z"]) == 0.0
    assert float(frame.loc[0, "z"]) == 10.0
