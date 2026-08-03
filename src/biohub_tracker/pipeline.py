from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from biohub_tracker.config import ProjectConfig
from biohub_tracker.coordinates import round_and_clip_voxel_zyx
from biohub_tracker.detection import detect_frame
from biohub_tracker.models import NodeIdAllocator, PredictedNode, PredictionGraph
from biohub_tracker.tracking import link_consecutive_nodes
from biohub_tracker.zarr_reader import VolumeDatasetReader


def run_prediction_pipeline(
    competition_root: str | Path,
    config: ProjectConfig,
) -> list[PredictionGraph]:
    """Detect cells and link consecutive frames on test volumes."""
    reader = VolumeDatasetReader(competition_root, split="test")
    graphs: list[PredictionGraph] = []
    for dataset in reader.dataset_names():
        metadata = reader.metadata(dataset)
        allocator = NodeIdAllocator(next_id=config.submission.node_id_start)
        nodes_by_t: dict[int, list[PredictedNode]] = defaultdict(list)
        for t in range(metadata.time_points):
            frame = reader.read_frame(dataset, t)
            detections = detect_frame(
                frame,
                dataset=dataset,
                t=t,
                voxel_spacing_zyx=metadata.voxel_spacing_zyx,
                config=config.detection,
            )
            for detection in detections:
                z, y, x = round_and_clip_voxel_zyx(
                    (detection.z, detection.y, detection.x),
                    metadata.spatial_shape_zyx,
                )
                nodes_by_t[t].append(
                    PredictedNode(
                        dataset=dataset,
                        node_id=allocator.allocate(),
                        t=t,
                        z=z,
                        y=y,
                        x=x,
                    )
                )
        nodes = [node for t in sorted(nodes_by_t) for node in nodes_by_t[t]]
        edges = []
        for t in range(1, metadata.time_points):
            edges.extend(
                link_consecutive_nodes(
                    nodes_by_t.get(t - 1, []),
                    nodes_by_t.get(t, []),
                    voxel_spacing_zyx=metadata.voxel_spacing_zyx,
                    config=config.tracking,
                )
            )
        graphs.append(PredictionGraph(nodes=nodes, edges=edges))
    return graphs
