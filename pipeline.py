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


def extract_path_permutations(
    image_path: str,
    display_highlight: bool,
    sm_threshold: int,
    depth: int,
    norm_cutoff: float,
    frac_length_cutoff: float,
    save: bool,
    overlay: bool,
    save_dir: str,
):
    binary_image = read_binary_image(image_path)

    skeleton = skeletonize_image(binary_image, "lee")

    branching_points, cval_data_dict = find_branch_points(skeleton)

    if display_highlight:
        hbeps_image = formatter.highlight_branch_points(skeleton, branching_points)
        plt.imshow(hbeps_image)
        plt.show()

    segments = segment.find_branch_segments(skeleton, branching_points)

    pair_data_arr = segment.find_segment_pair_labels(
        segments, cval_data_dict, sm_threshold
    )
    segment.extend_connected_branch_points(
        segments, pair_data_arr, branching_points, cval_data_dict, skeleton.shape
    )

    vector.calc_pairs_end_vectors(segments, branching_points, pair_data_arr, depth)

    neighbor_graph = path.construct_neighborhood_graph(pair_data_arr)

    # cutoff higher = more_smooth_path
    unique_paths = path.find_unique_paths(
        neighbor_graph, pair_data_arr, norm_cutoff, depth
    )

    unique_paths = path.prune_similar_paths(unique_paths, pair_data_arr, neighbor_graph)

    segment.segment_union(
        unique_paths,
        segments,
        pair_data_arr,
        skeleton,
        branching_points,
        frac_length_cutoff,
        save_dir,
        save,
        overlay,
    )


IMAGE_PATH = "test_images\processed_pdr (80)_0.jpg"

SM_THRESHOLD = 5  # px count to determine if segment is small
VECTOR_DEPTH = 5  # depth from the end of segments to calculate the vectors
NORM_CUTOFF = 0.75  # cosine of angle to cutoff if less than this value, range: (0,1)
FRAC_LENGTH_CUTOFF = 0.05  # fraction of width to determine if the segment union is too small to be included in the results, range: (0,1)
SAVE = False  # should save the results
SAVE_DIR = "res"  # save directory
SAVE_WITH_OVERLAY = True  # should show skeleton overlay

DISPLAY_HIGHLIGHTED_SKELETON = False

if __name__ == "__main__":
    extract_path_permutations(
        image_path=IMAGE_PATH,
        display_highlight=DISPLAY_HIGHLIGHTED_SKELETON,
        sm_threshold=SM_THRESHOLD,
        depth=VECTOR_DEPTH,
        norm_cutoff=NORM_CUTOFF,
        frac_length_cutoff=FRAC_LENGTH_CUTOFF,
        save=SAVE,
        overlay=SAVE_WITH_OVERLAY,
        save_dir=SAVE_DIR,
    )
