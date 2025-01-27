import numpy as np
import logging
from tqdm import tqdm
from vessel_analysis import get_path_length
from utils import edge


def find_paths_with_pruning(
    graph: dict[str, set],
    vessel_segments: dict[str, np.ndarray],
    points_dict: dict[str, dict],
    edge_lengths: dict[str, int],
    length_threshold: int,
    angle_threshold: float,
    minimum_final_length: int,
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
            edge.get_edge_length(node, nbr, edge_lengths) > length_threshold
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

                is_small = edge.is_small_segment(
                    node, nbr, length_threshold, edge_lengths
                )

                # if this is the first segment and it is small we don't include this path
                if len(curr_path) == 1 and is_small:
                    continue

                # if segment is not small
                if not is_small:
                    new_edge = " - ".join((node, nbr))

                    # if last edge exist compare new edge with last one
                    if last_edge is not None:
                        angle = edge.fit_angle_between_two_edges(
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

    _remove_small_final_path(valid_path_set, edge_lengths, minimum_final_length, logger)

    return list(
        sorted(
            valid_path_set,
            key=lambda x: (len(x), x),
        )
    )


def _remove_small_final_path(
    valid_path_set: set[tuple[str]],
    edge_lengths: dict[str, int],
    minimum_final_length: int,
    logger: logging.Logger,
) -> None:
    small_remove_count = 0
    path_tuple_list = list(valid_path_set)
    for path_tuple in tqdm(path_tuple_list, desc="Removing small final paths"):
        if len(path_tuple) > 1:
            if get_path_length(path_tuple, edge_lengths) <= minimum_final_length:
                valid_path_set.remove(path_tuple)
                small_remove_count += 1
    logger.info(
        "Total final small multilevel paths removed: " + str(small_remove_count)
    )
