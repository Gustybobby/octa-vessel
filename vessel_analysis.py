# vessel_analysis.py

import numpy as np
from collections import deque
from scipy.sparse import csr_matrix
import logging
from utils import edge

logger = logging.getLogger(__name__)


def get_length_of_vessel(segment: np.ndarray) -> int:
    """Calculates the length of a vessel segment."""
    logger.debug("Calculating vessel length")
    sparse_segment = csr_matrix(segment > 0)
    return sparse_segment.nnz


def find_direct_path(
    skeleton_image: np.ndarray,
    start_label: str,
    end_label: str,
    branching_points: list[tuple[int, int]],
    num_points: int,
) -> tuple[np.ndarray, dict[str, list[tuple[int, int]]]]:
    """
    Finds the direct path between two points in the skeleton image using BFS.
    """
    logger.debug(f"Finding direct path from {start_label} to {end_label}")
    start_coord = branching_points[int(start_label)]
    end_coord = branching_points[int(end_label)]

    branch_coords_set = set(branching_points)
    h, w = skeleton_image.shape
    segment_image = np.zeros_like(skeleton_image)
    queue = deque([(start_coord, [start_coord])])
    visited = set([start_coord])
    path_points = []

    while queue:
        (x, y), path = queue.popleft()
        path_points = path

        if (x, y) == end_coord:
            for px, py in path:
                segment_image[py, px] = 255
            break

        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                neighbor = (nx, ny)
                if (
                    0 <= nx < w
                    and 0 <= ny < h
                    and skeleton_image[ny, nx] == 1
                    and neighbor not in visited
                ):

                    if neighbor in branch_coords_set and neighbor != end_coord:
                        continue

                    queue.append((neighbor, path + [neighbor]))
                    visited.add(neighbor)

    points_info = {
        start_label: path_points[:num_points],
        end_label: path_points[-num_points:],
    }

    logger.debug(f"Direct path found with length {len(path_points)}")
    return segment_image, points_info


def extract_vessel_segments(
    single_level_paths: list[tuple[str, str]],
    skeleton_image: np.ndarray,
    branching_points: list[tuple[int, int]],
    num_points: int,
) -> tuple[dict[str, np.ndarray], dict[str, dict]]:
    """
    Extracts vessel segments and their corresponding points.
    """
    logger.info("Extracting vessel segments")
    vessel_segments = {}
    points = {}

    for path in single_level_paths:
        start_label, end_label = path
        segment_image, points_info = find_direct_path(
            skeleton_image, start_label, end_label, branching_points, num_points
        )

        segment_key = f"{start_label} - {end_label}"
        vessel_segments[segment_key] = segment_image
        points[segment_key] = points_info

    logger.info(f"Extracted {len(vessel_segments)} vessel segments")
    return vessel_segments, points


def find_single_level_paths(graph: dict[str, set]):
    single_level_paths = []
    for u in graph:
        for v in graph[u]:
            if u < v:
                single_level_paths.append((u, v))
    return single_level_paths


def get_path_length(path_tuple, edge_lengths):
    path_length = 0
    for i in range(len(path_tuple) - 1):
        path_length += edge.get_edge_length(
            path_tuple[i], path_tuple[i + 1], edge_lengths
        )
    return path_length


def calculate_TI(branching_points, edge_lengths, path_tuple):

    path_length = get_path_length(path_tuple, edge_lengths)

    start = branching_points[int(path_tuple[0])]
    end = branching_points[int(path_tuple[-1])]

    return path_length / np.sqrt((start[0] - end[0]) ** 2 + (start[1] - end[1]) ** 2)


def find_edges_lengths_and_tortuosities(
    vessel_segments: dict[str, np.ndarray], branching_points: list[tuple[int, int]]
) -> tuple[dict[str, int], dict[str, np.float64]]:

    edge_lengths: dict[str, int] = {}

    # path tuple : TI score (actual length / euclidean distance)
    tortuosity_index: dict[str, np.float64] = {}

    for edge_key, segment_img in vessel_segments.items():
        edge_lengths[edge_key] = get_length_of_vessel(segment_img)

        path_tuple = edge_key.split(" - ")
        tortuosity_index[edge_key] = calculate_TI(
            branching_points, edge_lengths, path_tuple
        )

    return edge_lengths, tortuosity_index
