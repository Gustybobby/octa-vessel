import numpy as np

from graph.node import Node
from graph.build_valid_paths import get_path_length


def calculate_TI(
    path: tuple[str],
    coord_to_nodes: dict[tuple[int, int], Node],
    segments: dict[tuple[str, str], set[tuple[int, int]]],
):
    id_to_nodes: dict[str, Node] = {}
    for node in coord_to_nodes.values():
        id_to_nodes[node.id] = node

    path_length = get_path_length(path, segments)

    start = id_to_nodes[path[0]]
    end = id_to_nodes[path[-1]]

    return path_length / np.sqrt((start.x - end.x) ** 2 + (start.y - end.y) ** 2)
