# graph_analysis.py

import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage import convolve, binary_hit_or_miss
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


def find_branch_edge_points(
    skeleton_image: np.ndarray, method: str = "MATRIX_DEGREE", display_beps=False
) -> list[tuple[int, int]]:
    """Identifies branch points and edge points in the skeleton image. method = ("SELEMS","MATRIX_DEGREE")"""
    logger.info("Finding branch and edge points in the skeleton")

    kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)
    neighbors = convolve(skeleton_image, kernel, mode="constant", cval=0)

    if method == "SELEMS":
        # this method scan throughs branching point templates
        selems = list()
        selems.append(np.array([[0, 1, 0], [1, 1, 1], [0, 0, 0]]))
        selems.append(np.array([[1, 0, 1], [0, 1, 0], [1, 0, 0]]))
        selems.append(np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0]]))
        selems.append(np.array([[0, 1, 0], [1, 1, 0], [0, 0, 1]]))
        selems.append(np.array([[0, 0, 1], [1, 1, 1], [0, 1, 0]]))
        selems = [np.rot90(selems[i], k=j) for i in range(5) for j in range(4)]
        selems.append(np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]))
        selems.append(np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]]))

        bp_image = np.zeros_like(skeleton_image, dtype=bool)
        for selem in selems:
            bp_image |= binary_hit_or_miss(skeleton_image, selem)

    elif method == "MATRIX_DEGREE":
        # this method look for neighborhood with matrix degree > 2
        bp_image = (skeleton_image == 1) & (neighbors > 2)
        bp_image = bp_image.astype(np.uint8)

    else:
        raise Exception("invalid method")

    edge_image = (skeleton_image == 1) & (neighbors == 1)
    edge_image = edge_image.astype(np.uint8)

    beps = []
    h, w = bp_image.shape
    for y in range(h):
        for x in range(w):
            isBp = (method == "SELEMS" and bp_image[y][x] == True) or (
                method != "SELEMS" and bp_image[y, x] == 1
            )
            if isBp or edge_image[y, x] == 1:
                beps.append((x, y))

    logger.info(f"Found {len(beps)} branch and edge points")

    if display_beps:
        bp_skel = skeleton_image.copy()
        for x, y in beps:
            bp_skel[y, x] += 2
        plt.imshow(bp_skel)
        plt.show()

    return beps


def find_neighborhood_graph(
    skeleton_image: np.ndarray, beps: list[tuple[int, int]]
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
