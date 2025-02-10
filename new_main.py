from image_processing.read_binary_image import read_binary_image
from image_processing.skeletonize_image import skeletonize_image
from image_processing.build_distinct_images import build_distinct_images

from graph.find_nodes import find_nodes
from graph.construct_nodes_neighborhood import construct_nodes_neighborhood
from graph.extract_segments import extract_segments
from graph.prune_small_branches import prune_small_branches
from graph.build_valid_paths import build_valid_paths

import config
import visualization
import vessel_analysis
import os


def main(image_path: str, result_dir: str):
    bin_image = read_binary_image(image_path)
    skeleton = skeletonize_image(bin_image, method="lee")

    coord_to_nodes = find_nodes(skeleton)
    construct_nodes_neighborhood(coord_to_nodes, skeleton)

    segments = extract_segments(coord_to_nodes, skeleton)
    prune_small_branches(coord_to_nodes, segments, config.SMALL_BRANCH_THRESHOLD)

    valid_paths = build_valid_paths(
        coord_to_nodes,
        segments,
        config.SMALL_SEGMENT_THRESHOLD,
        config.NUM_POINTS,
        config.ANGLE_THRESHOLD,
        config.MIN_FINAL_LENGTH,
        config.MAX_DEPTH,
    )

    valid_paths_images = build_distinct_images(
        valid_paths, segments, config.MSE_THRESHOLD, skeleton.shape
    )

    tortuosity_index = {}
    for path in valid_paths_images:
        tortuosity_index[path] = vessel_analysis.calculate_TI(
            path, coord_to_nodes, segments
        )

    visualization.save_images(
        valid_paths_images,
        result_dir,
        config.OVERLAY_IMAGE,
        config.OVERLAY_INTENSITY,
        skeleton,
        tortuosity_index,
        key=lambda x: tuple([int(n) for n in x]),
    )


if __name__ == "__main__":
    DATASET_DIR = "Dataset/TV_TUH_processed"
    for filename in os.listdir(DATASET_DIR):
        main(
            os.path.join(DATASET_DIR, filename),
            os.path.join("extraction", DATASET_DIR, filename.split(".")[0], "result"),
        )
