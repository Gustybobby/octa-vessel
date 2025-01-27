import numpy as np
from skimage.morphology import skeletonize
import logging


def skeletonize_image(
    image: np.ndarray, method: str = "lee", logger: logging.Logger = None
) -> np.ndarray:
    """Skeletonizes the binary image using the specified method."""

    logger.info(f"Skeletonizing image using method: {method}")

    skel = skeletonize(image // 255, method=method)
    skel = skel.astype(np.uint8)

    logger.info("Image skeletonization complete")

    return skel
