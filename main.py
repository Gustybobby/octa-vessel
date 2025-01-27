# main.py

import logging

import numpy as np
from tqdm import tqdm

from config import (
    LOG_LEVEL,
    LOG_FORMAT,
    RESULT_DIR,
    INPUT_IMAGE,
    OVERLAY_IMAGE,
    OVERLAY_INTENSITY,
    ANGLE_THRESHOLD,
    MAX_RECURSION_DEPTH,
    SMALL_SEGMENT_LENGTH,
    NUM_POINTS,
    MINIMUM_FINAL_LENGTH,
    MSE_THRESHOLD,
)
import image_processing
import graph_analysis
import vessel_analysis
import utilities
import visualization
import image_similarity_remove


def setup_logging():
    """Configures the logging settings."""
    logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
    logger = logging.getLogger()
    return logger


def get_edge_key(a: str, b: str) -> str:
    return f"{min(a, b)} - {max(a, b)}"


def get_edge_length(a: str, b: str, edge_lengths: dict[str, int]) -> int:
    return edge_lengths.get(get_edge_key(a, b), 0)


def fit_angle_between_two_edges(
    edge1: str, edge2: str, points_dict: dict, vessel_segments
) -> float:
    return utilities.fit_tangents_at_junction(
        points_dict, edge1, edge2, vessel_segments, NUM_POINTS
    )


def is_small_segment(
    u: str, v: str, threshold: int, edge_lengths: dict[str, int]
) -> bool:
    return get_edge_length(u, v, edge_lengths) <= threshold


def find_all_paths_with_pruning(
    graph: dict[str, set],
    vessel_segments: dict[str, np.ndarray],
    points_dict: dict[str, dict],
    edge_lengths: dict[str, int],
    length_threshold: int,
    angle_threshold: float,
    logger: logging.Logger,
    max_length: int = None,
) -> list[tuple[str]]:
    """
    Enumerates all paths in 'graph' (iterative DFS), pruning based on constraints.
    """
    logger.info("Starting single-pass DFS with on-the-fly pruning...")
    valid_path_set = set()
    valid_starts = set()
    for node in graph:
        if any(
            get_edge_length(node, nbr, edge_lengths) > length_threshold
            for nbr in graph[node]
        ):
            valid_starts.add(node)
    valid_starts = sorted(list(valid_starts), key=int)

    for start_node in tqdm(valid_starts, desc="DFS from valid_starts"):

        visited = set([start_node])
        curr_path = [start_node]

        def backtrack(node: str, last_edge: str | None):
            # add curr path to valid path
            if len(curr_path) > 2:
                is_reversed = tuple(reversed(curr_path)) in valid_path_set
                last_is_small = last_edge != " - ".join([curr_path[-2], curr_path[-1]])

                if not is_reversed and not last_is_small:
                    valid_path_set.add(tuple(curr_path))

            # if curr path exceed max length
            if max_length is not None and len(curr_path) >= max_length:
                return

            for nbr in sorted(list(graph[node]), key=int):
                if nbr in visited:
                    continue

                is_small = is_small_segment(node, nbr, length_threshold, edge_lengths)

                # if this is the first segment and it is small we don't include this path
                if len(curr_path) == 1 and is_small:
                    continue

                # if segment is not small
                if not is_small:
                    new_edge = " - ".join((node, nbr))

                    # if last edge exist compare new edge with last one
                    if last_edge is not None:
                        angle = fit_angle_between_two_edges(
                            last_edge, new_edge, points_dict, vessel_segments
                        )
                        if angle > angle_threshold:
                            continue

                visited.add(nbr)
                curr_path.append(nbr)

                backtrack(nbr, last_edge if is_small else new_edge)

                visited.remove(nbr)
                curr_path.pop()

        backtrack(start_node, None)

    logger.info("Total paths: " + str(len(valid_path_set)))

    # remove vessels with lesser than final size
    small_remove_count = 0
    path_tuple_list = list(valid_path_set)
    for path_tuple in tqdm(path_tuple_list, desc="Removing small final paths"):
        if len(path_tuple) > 1:
            if get_path_length(path_tuple, edge_lengths) <= MINIMUM_FINAL_LENGTH:
                valid_path_set.remove(path_tuple)
                small_remove_count += 1
    logger.info(
        "Total final small multilevel paths removed: " + str(small_remove_count)
    )

    return list(
        sorted(
            valid_path_set,
            key=lambda x: (len(x), x),
        )
    )


def get_path_length(path_tuple, edge_lengths):
    path_length = 0
    for i in range(len(path_tuple) - 1):
        path_length += get_edge_length(path_tuple[i], path_tuple[i + 1], edge_lengths)
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
        edge_lengths[edge_key] = vessel_analysis.get_length_of_vessel(segment_img)

        path_tuple = edge_key.split(" - ")
        tortuosity_index[edge_key] = calculate_TI(
            branching_points, edge_lengths, path_tuple
        )

    return edge_lengths, tortuosity_index


def remove_small_leaves_from_graph(
    edge_lengths: dict[str, int],
    graph: dict[str, set],
    leaves_threshold: int,
    logger: logging.Logger,
):
    nodes_to_remove = set()
    for node in graph:
        if len(graph[node]) > 1:
            continue
        nbr = list(graph[node])[0]
        if get_edge_length(node, nbr, edge_lengths) < leaves_threshold:
            nodes_to_remove.add(node)
            edge_key = get_edge_key(node, nbr)
            if edge_key in edge_lengths:
                edge_lengths.pop(get_edge_key(node, nbr))

    graph_nodes = list(graph.keys())
    for node in graph_nodes:
        if node not in graph:
            continue
        if node in nodes_to_remove:
            graph.pop(node)
        else:
            graph[node] = graph[node] - nodes_to_remove
        if len(graph[node]) == 0:
            graph.pop(node)

    logger.info("Total small leaves removed: " + str(len(nodes_to_remove)))

    return len(nodes_to_remove)


# we do this to check MSE on the fly because images takes up memory space (150k+ images cannot fit within 32 GB memory)
def build_distinct_images(
    valid_path_list: list[tuple[str]],
    vessel_segments: dict[str, np.ndarray],
    mse_threshold: float,
    logger: logging.Logger,
):
    valid_paths = {}
    downsampled_images = {}
    for path_tuple in tqdm(valid_path_list, desc="Building images"):
        segments = [
            vessel_segments[get_edge_key(path_tuple[i], path_tuple[i + 1])].astype(
                np.uint8
            )
            for i in range(len(path_tuple) - 1)
        ]
        image = np.maximum.reduce(segments)

        downsampled = image_similarity_remove.downsample_image(image, 256)

        isDupe = False

        # check with existing distinct path tuples
        for downsampled_path_tuple in downsampled_images:
            # compare downsampled candidate with existing downsampled
            error = image_similarity_remove.mse(
                downsampled_images[downsampled_path_tuple], downsampled
            )
            # if too similar, set dupe to True then break
            if error <= mse_threshold:
                isDupe = True
                break

        if not isDupe:
            valid_paths[path_tuple] = image
            downsampled_images[path_tuple] = downsampled

    logger.info(f"Total distinct images: {len(valid_paths)}")

    return valid_paths


def main():
    logger = setup_logging()
    logger.info("Starting vessel analysis pipeline")

    # Step 1: Read and preprocess the image
    binary_image = image_processing.read_binary_image(INPUT_IMAGE)
    skeleton = image_processing.skeletonize_image(binary_image, method="lee")

    # Step 2: Find branch and edge points
    branching_points = graph_analysis.find_branch_edge_points(
        skeleton, display_beps=False
    )

    # Step 3: Create neighborhood graph
    neighbor_graph, info_dict = graph_analysis.find_neighborhood_graph(
        skeleton, branching_points
    )

    # Step 4: Extract vessel segments
    single_level_paths = []
    for A in neighbor_graph:
        for B in neighbor_graph[A]:
            if A < B:
                single_level_paths.append((A, B))

    vessel_segments, points = vessel_analysis.extract_vessel_segments(
        single_level_paths, skeleton, branching_points, num_points=NUM_POINTS
    )

    # Step 5: Precompute edge lengths (We calculate TI as well)
    edge_lengths, tortuosity_index = find_edges_lengths_and_tortuosities(
        vessel_segments, branching_points
    )

    while remove_small_leaves_from_graph(edge_lengths, neighbor_graph, 10, logger) > 0:
        continue

    # Execute path finding
    valid_path_list = find_all_paths_with_pruning(
        neighbor_graph,
        vessel_segments,
        points,
        edge_lengths,
        SMALL_SEGMENT_LENGTH,
        ANGLE_THRESHOLD,
        logger,
        max_length=MAX_RECURSION_DEPTH,
    )

    valid_paths = build_distinct_images(
        valid_path_list, vessel_segments, mse_threshold=MSE_THRESHOLD, logger=logger
    )

    for path_tuple in valid_paths:
        tortuosity_index[path_tuple] = calculate_TI(
            branching_points, edge_lengths, path_tuple
        )

    # Step 8: Save the valid paths images
    visualization.save_images(
        "multi",
        valid_paths,
        RESULT_DIR,
        OVERLAY_IMAGE,
        OVERLAY_INTENSITY,
        skeleton,
        tortuosity_index,
        key=lambda x: tuple([int(n) for n in x]),
    )
    visualization.save_images(
        "single",
        vessel_segments,
        RESULT_DIR,
        OVERLAY_IMAGE,
        OVERLAY_INTENSITY,
        skeleton,
        tortuosity_index,
        lambda a, b: get_edge_length(a, b, edge_lengths),
    )
    logger.info("Vessel analysis pipeline completed successfully")


if __name__ == "__main__":
    main()
