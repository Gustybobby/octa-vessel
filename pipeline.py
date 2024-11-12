from skimage.morphology import skeletonize
from scipy.ndimage import convolve
from collections import defaultdict
import numpy as np
import cv2
import logging
import matplotlib.pyplot as plt
import segment
import vector
import path
import formatter

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


def skeletonize_image(image, method="lee"):
    skel = skeletonize(image // 255, method=method)
    skel = skel.astype(np.uint8)

    logging.info("skeletonized image")
    return skel


def find_branch_points(skeleton_image) -> tuple[list[tuple[int, int]], dict]:
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

    cval_to_label = {}
    for i, (x, y) in enumerate(beps):
        coord_value = x * h + y
        cval_to_label[coord_value] = {"label": str(i), "coord": (x, y)}

    logging.info("found " + str(len(beps)) + " branch edge points")
    return beps, cval_to_label


def construct_neighborhood_graph(
    pair_label: list[list[dict | None]],
) -> dict[str, set[str]]:
    graph = defaultdict(set)
    for l1 in range(len(pair_label)):
        for l2 in range(len(pair_label[l1])):
            if pair_label[l1][l2]:
                graph[str(l1)].add(str(l2))
                graph[str(l2)].add(str(l1))

    return graph


if __name__ == "__main__":
    binary_image = read_binary_image("test_images\processed_pdr (71)_0.jpg")

    skeleton = skeletonize_image(binary_image, "lee")

    branching_points, cval_label_dict = find_branch_points(skeleton)

    hbeps_image = formatter.highlight_branch_points(skeleton, branching_points)
    plt.imshow(hbeps_image)
    plt.show()

    segments = segment.find_branch_segments(skeleton, branching_points)

    pair_label_arr = segment.find_segment_pair_labels(
        segments, cval_label_dict, sm_threshold=5
    )
    segment.extend_connected_branch_points(
        segments, pair_label_arr, branching_points, cval_label_dict, skeleton.shape
    )

    vector.calc_pairs_end_vectors(segments, branching_points, pair_label_arr, depth=10)

    neighbor_graph = construct_neighborhood_graph(pair_label_arr)

    # cutoff higher = more_smooth_path
    unique_paths = path.find_unique_paths(
        neighbor_graph, pair_label_arr, norm_cutoff=0.707
    )

    last_unique_len = 0
    while last_unique_len != len(unique_paths):
        last_unique_len = len(unique_paths)
        unique_paths = path.prune_similar_paths(unique_paths, pair_label_arr)

    segment.segment_union(
        unique_paths,
        segments,
        pair_label_arr,
        skeleton,
        branching_points,
        save=True,
        overlay=False,
    )
