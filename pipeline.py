import numpy as np
import matplotlib.pyplot as plt
import cv2
import logging
import segment
import vector
import path
import formatter
from skimage.morphology import skeletonize
from scipy.ndimage import convolve
from classes.branch_point_data import BranchPointData

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)


def read_binary_image(filename: str):
    image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    _, image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    logging.info("loaded " + filename)
    return image


def skeletonize_image(image, method="lee") -> np.ndarray:
    skel = skeletonize(image // 255, method=method)
    skel = skel.astype(np.uint8)

    logging.info("skeletonized image")
    return skel


def find_branch_points(
    skeleton_image: np.ndarray,
) -> tuple[list[tuple[int, int]], dict[int, BranchPointData]]:
    kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)
    neighbors = convolve(skeleton_image, kernel, mode="constant", cval=0)

    bp_image = (skeleton_image == 1) & (neighbors > 2)
    bp_image = bp_image.astype(np.uint8)

    edge_image = (skeleton_image == 1) & (neighbors == 1)
    edge_image = edge_image.astype(np.uint8)

    beps = []
    h, w = bp_image.shape
    for y in range(h):
        for x in range(w):
            if bp_image[y, x] == 1 or edge_image[y, x] == 1:
                beps.append((x, y))

    cval_to_data = {}
    for i, (x, y) in enumerate(beps):
        coord_value = x * h + y
        cval_to_data[coord_value] = BranchPointData(label=str(i), coord=(x, y))

    logging.info("found " + str(len(beps)) + " branch edge points")
    return beps, cval_to_data


if __name__ == "__main__":
    binary_image = read_binary_image("test_images\processed_pdr (114)_0.jpg")

    skeleton = skeletonize_image(binary_image, "lee")

    branching_points, cval_data_dict = find_branch_points(skeleton)

    hbeps_image = formatter.highlight_branch_points(skeleton, branching_points)
    plt.imshow(hbeps_image)
    plt.show()

    segments = segment.find_branch_segments(skeleton, branching_points)

    pair_data_arr = segment.find_segment_pair_labels(
        segments, cval_data_dict, sm_threshold=5
    )
    segment.extend_connected_branch_points(
        segments, pair_data_arr, branching_points, cval_data_dict, skeleton.shape
    )

    vector.calc_pairs_end_vectors(segments, branching_points, pair_data_arr, depth=5)

    neighbor_graph = path.construct_neighborhood_graph(pair_data_arr)

    # cutoff higher = more_smooth_path
    unique_paths = path.find_unique_paths(
        neighbor_graph, pair_data_arr, norm_cutoff=0.75, depth=5
    )

    unique_paths = path.prune_similar_paths(unique_paths, pair_data_arr, neighbor_graph)

    segment.segment_union(
        unique_paths,
        segments,
        pair_data_arr,
        skeleton,
        branching_points,
        save=True,
        overlay=True,
    )
