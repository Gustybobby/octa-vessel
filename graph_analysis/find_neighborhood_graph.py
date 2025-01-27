import numpy as np
import logging
from collections import defaultdict


def find_neighborhood_graph(
    skeleton_image: np.ndarray, beps: list[tuple[int, int]], logger: logging.Logger
) -> tuple[dict[str, set], dict[int, dict]]:
    """Creates a neighborhood graph based on branch edge points in the skeleton."""
    logger.info("Creating neighborhood graph from branch and edge points")
    h, w = skeleton_image.shape

    bp_info_map = {}
    for i, (x, y) in enumerate(beps):
        coord_value = x * h + y  # Unique coordinate value
        bp_info_map[coord_value] = {"label": str(i), "coord": (x, y)}

    bp_coord_value_set = set([x * h + y for (x, y) in beps])

    graph = defaultdict(set)
    for x, y in beps:
        visited = set()
        stack = [(x, y)]

        coord_value_start = x * h + y
        start_node: str = bp_info_map[coord_value_start]["label"]

        while stack:
            x_curr, y_curr = stack.pop()

            if x_curr < 0 or y_curr < 0 or x_curr >= w or y_curr >= h:
                continue

            coord_value_current = x_curr * h + y_curr
            not_path = skeleton_image[y_curr, x_curr] == 0

            if not_path or coord_value_current in visited:
                continue
            visited.add(coord_value_current)

            not_start = x_curr != x or y_curr != y
            is_bpe = coord_value_current in bp_coord_value_set

            if not_start and is_bpe:
                neighbor_node: str = bp_info_map[coord_value_current]["label"]
                graph[start_node].add(neighbor_node)
                continue

            # Add all neighbors to the stack for traversal
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    stack.append((x_curr + dx, y_curr + dy))

    logger.info(f"Neighborhood graph created with {len(graph)} nodes")

    return graph, bp_info_map
