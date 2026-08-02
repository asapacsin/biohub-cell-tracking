from biohub_tracker.models import NodeIdAllocator, PredictedEdge, PredictedNode


def test_node_ids_remain_unique_across_frames() -> None:
    allocator = NodeIdAllocator()
    nodes = [PredictedNode("a", allocator.allocate(), t, 0, 0, 0) for t in range(4)]
    assert [node.node_id for node in nodes] == [1, 2, 3, 4]


def test_node_ids_may_restart_for_another_dataset() -> None:
    first = NodeIdAllocator()
    second = NodeIdAllocator()
    assert [first.allocate(), first.allocate()] == [1, 2]
    assert [second.allocate(), second.allocate()] == [1, 2]


def test_same_biological_track_uses_distinct_detection_nodes() -> None:
    allocator = NodeIdAllocator()
    before = PredictedNode("a", allocator.allocate(), 0, 2, 3, 4)
    after = PredictedNode("a", allocator.allocate(), 1, 2, 4, 4)
    edge = PredictedEdge("a", before.node_id, after.node_id)
    assert before.node_id != after.node_id
    assert edge == PredictedEdge("a", 1, 2)

