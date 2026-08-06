from __future__ import annotations

import pandas as pd

from biohub_pipeline.evaluation import official_spec_evaluate, official_spec_per_sample


def test_official_spec_lite_perfect_tiny_chain() -> None:
    nodes = pd.DataFrame(
        [
            {"node_id": 1, "t": 0, "z": 1, "y": 1, "x": 1},
            {"node_id": 2, "t": 1, "z": 1, "y": 2, "x": 1},
        ]
    )
    edges = pd.DataFrame([{"source_id": 1, "target_id": 2}])
    result = official_spec_evaluate(nodes, edges, nodes, edges)
    assert (result.edge_tp, result.edge_fp, result.edge_fn) == (1, 0, 0)
    scores = official_spec_per_sample(result, estimated_total_nodes=2)
    assert scores["adj_edge_jaccard"] == 1.0
    assert scores["node_recall"] == 1.0
