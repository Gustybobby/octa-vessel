from graph.node import Node


def prune_small_branches(
    coord_to_nodes: dict[tuple[int, int], Node],
    segments: dict[tuple[str, str], set[tuple[int, int]]],
    threshold: int,
):
    print("[BRANCH_PRUNING] Start pruning small branches")

    while _prune_step(coord_to_nodes, segments, threshold) > 0:
        continue

    print("[BRANCH_PRUNING] Final nodes count:", len(coord_to_nodes))
    print("[BRANCH_PRUNING] Final segments count:", len(segments))
    return


def _prune_step(
    coord_to_nodes: dict[tuple[int, int], Node],
    segments: dict[tuple[str, str], set[tuple[int, int]]],
    threshold: int,
):
    prune_count = 0
    nodes = list(coord_to_nodes.values())
    for node in nodes:
        if len(node.get_neighbors()) == 1:
            nbr = node.get_neighbors()[0]

            path_length = len(segments[node.get_segment_key(nbr)])

            if path_length < threshold:
                coord_to_nodes.pop((node.x, node.y))
                nbr.neighbors.pop(node.id)
                segments.pop(node.get_segment_key(nbr))
                prune_count += 1

    nodes = list(coord_to_nodes.values())
    for node in nodes:
        if len(node.get_neighbors()) == 0:
            coord_to_nodes.pop((node.x, node.y))

    return prune_count
