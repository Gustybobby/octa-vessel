import numpy as np
import logging
from scipy.ndimage import convolve
from matplotlib import pyplot as plt


def find_branch_edge_points(
    skeleton_image: np.ndarray, logger: logging.Logger, display_beps=False
) -> list[tuple[int, int]]:
    """Identifies branch points and edge points in the skeleton image."""

    logger.info("Finding branch and edge points in the skeleton")

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

    logger.info(f"Found {len(beps)} branch and edge points")

    if display_beps:
        _display_beps_overlayed_skeleton(beps, skeleton_image)

    return beps


def _display_beps_overlayed_skeleton(
    beps: list[tuple[int, int]], skeleton_image: np.ndarray
) -> None:
    bp_skel = skeleton_image.copy()
    for x, y in beps:
        bp_skel[y, x] += 1
    plt.imshow(bp_skel)
    plt.show()
