# main.py

import logging
from config import (
    LOG_LEVEL,
    LOG_FORMAT,
    RESULT_DIR,
    INPUT_IMAGE,
    OVERLAY_IMAGE,
    OVERLAY_INTENSITY,
    SHOW_PRUNED_IMAGE,
    ANGLE_THRESHOLD,
    MAX_RECURSION_DEPTH,
    SMALL_SEGMENT_LENGTH,
    NUM_POINTS,
    MINIMUM_FINAL_LENGTH,
    LEAF_BRANCH_LENGTH,
    MSE_THRESHOLD,
)
from image_processing.read_binary_image import read_binary_image
from image_processing.skeletonize_image import skeletonize_image
from image_processing.build_distinct_images import build_distinct_images
from graph_analysis.find_branch_edge_points import find_branch_edge_points
from graph_analysis.find_neighborhood_graph import find_neighborhood_graph
from graph_analysis.remove_small_leaves import remove_small_leaves
from graph_analysis.find_path_with_pruning import find_paths_with_pruning
from utils.edge import get_edge_length
import vessel_analysis
import visualization
from matplotlib import pyplot as plt
import numpy


def setup_logging():
    """Configures the logging settings."""
    logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
    logger = logging.getLogger()
    return logger


def main():
    logger = setup_logging()
    logger.info("Starting vessel analysis pipeline")

    # Step 1: Read and preprocess the image
    binary_image = read_binary_image(INPUT_IMAGE, logger)
    skeleton = skeletonize_image(binary_image, method="lee", logger=logger)

    # Step 2: Find branch and edge points
    branching_points = find_branch_edge_points(skeleton, logger, display_beps=False)

    # Step 3: Create neighborhood graph
    neighbor_graph, info_dict = find_neighborhood_graph(
        skeleton, branching_points, logger
    )

    # Step 4: Extract vessel segments
    single_level_paths = vessel_analysis.find_single_level_paths(neighbor_graph)

    vessel_segments, points = vessel_analysis.extract_vessel_segments(
        single_level_paths, skeleton, branching_points, num_points=NUM_POINTS
    )

    # Step 5: Precompute edge lengths (We calculate TI as well)
    edge_lengths, tortuosity_index = (
        vessel_analysis.find_edges_lengths_and_tortuosities(
            vessel_segments, branching_points
        )
    )

    while (
        remove_small_leaves(edge_lengths, neighbor_graph, LEAF_BRANCH_LENGTH, logger)
        > 0
    ):
        continue

    if SHOW_PRUNED_IMAGE:
        pruned_image = numpy.maximum.reduce(
            [vessel_segments[edge] for edge in edge_lengths]
        )
        for x, y in branching_points:
            if info_dict[x * pruned_image.shape[0] + y]["label"] in neighbor_graph:
                pruned_image[y, x] += 128
        plt.imshow(pruned_image)
        plt.show()

    # Execute path finding
    valid_path_list = find_paths_with_pruning(
        neighbor_graph,
        vessel_segments,
        points,
        edge_lengths,
        SMALL_SEGMENT_LENGTH,
        ANGLE_THRESHOLD,
        MINIMUM_FINAL_LENGTH,
        logger,
        MAX_RECURSION_DEPTH,
    )

    valid_paths = build_distinct_images(
        valid_path_list, vessel_segments, mse_threshold=MSE_THRESHOLD, logger=logger
    )

    for path_tuple in valid_paths:
        tortuosity_index[path_tuple] = vessel_analysis.calculate_TI(
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
