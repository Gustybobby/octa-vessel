from graph.node import Node
import numpy as np
import heapq
import math
import matplotlib.pyplot as plt
from tqdm import tqdm


def get_segment_key(a: str, b: str) -> tuple[str, str]:
    return (a, b) if int(a) < int(b) else (b, a)


def extract_segments(
    coord_to_nodes: dict[tuple[int, int], Node], skeleton: np.ndarray
) -> dict[tuple[str, str], set[tuple[int, int]]]:
    print("[SEGMENT_EXTRACT] Extracting vessel segments")

    segments = {}
    for node in tqdm(coord_to_nodes.values(), desc="Vessel segments extraction"):
        for nb in node.neighbors.values():
            if (node.id, nb.id) in segments or (nb.id, node.id) in segments:
                continue

            sorted_node_id_pair = (
                (node.id, nb.id) if int(node.id) < int(nb.id) else (nb.id, node.id)
            )
            segments[sorted_node_id_pair] = set(
                _extract_path(node, nb, coord_to_nodes, skeleton)
            )

    print(f"[SEGMENT_EXTRACT] Extracted {len(segments)} segments")

    return segments


def plot_colored_segments(
    segments: dict[tuple[str, str], set[tuple[int, int]]],
    skeleton: np.ndarray,
    title: str = "",
):
    colored = np.zeros(skeleton.shape) + skeleton * 0.5

    color = 0
    for path in segments.values():
        for x, y in path:
            colored[y, x] = color + 1
        color = (color + 1) % 4

    plt.title(title)
    plt.imshow(colored)
    plt.show()


def _extract_path(
    node1: Node,
    node2: Node,
    coord_to_nodes: dict[tuple[int, int], Node],
    skeleton: np.ndarray,
) -> tuple[tuple[int, int]]:
    """
    Finds the direct path between two nodes in the skeleton using A*.
    """

    def heuristic(x1, y1) -> float:
        return math.sqrt((x1 - node2.x) ** 2 + (y1 - node2.y) ** 2)

    start = (node1.x, node1.y)
    heap = [(heuristic(node1.x, node1.y), 0, start, [start])]
    visited = set()

    h, w = skeleton.shape
    while heap:
        _, move, curr, path = heapq.heappop(heap)
        curr_x, curr_y = curr

        if curr_x < 0 or curr_y < 0 or curr_x >= w or curr_y >= h:
            continue

        if skeleton[curr_y, curr_x] == 0:
            continue

        if curr in visited:
            continue
        visited.add(curr)

        if curr in coord_to_nodes and curr != start:
            if coord_to_nodes[curr].id == node2.id:
                return path
            continue

        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                new_x, new_y = curr_x + dx, curr_y + dy
                heapq.heappush(
                    heap,
                    (
                        heuristic(new_x, new_y),
                        move + 1,
                        (new_x, new_y),
                        path.copy() + [(new_x, new_y)],
                    ),
                )

    raise Exception("Path not found")
