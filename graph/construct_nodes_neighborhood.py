from graph.node import Node
import numpy as np


def construct_nodes_neighborhood(
    coord_to_nodes: dict[tuple[int, int], Node], skeleton: np.ndarray
) -> None:
    """Creates a neighborhood based on nodes in the skeleton."""
    for node in coord_to_nodes.values():
        _add_adjacent_nodes(node, coord_to_nodes, skeleton)


def _add_adjacent_nodes(
    start_node: Node, coord_to_nodes: dict[tuple[int, int], Node], skeleton: np.ndarray
):
    start_coord = (start_node.x, start_node.y)
    stack = [start_coord]
    visited = set()

    h, w = skeleton.shape

    while stack:
        curr = stack.pop()
        curr_x, curr_y = curr

        if curr_x < 0 or curr_y < 0 or curr_x >= w or curr_y >= h:
            continue

        if skeleton[curr_y, curr_x] == 0:
            continue

        if curr in visited:
            continue
        visited.add(curr)

        if curr in coord_to_nodes and curr != start_coord:
            start_node.add_neighbor(coord_to_nodes[curr])
            continue

        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                stack.append((curr_x + dx, curr_y + dy))
