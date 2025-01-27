# image_processing.py

import cv2
import numpy as np
from skimage.morphology import skeletonize
import logging

logger = logging.getLogger(__name__)


def read_binary_image(filename: str) -> np.ndarray:
    """Reads an image from the specified filename and converts it to a binary image."""
    logger.info(f"Reading image from {filename}")
    image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    if image is None:
        logger.error(f"Failed to read image from {filename}")
        raise FileNotFoundError(f"Image file '{filename}' not found.")
    _, image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    logger.info("Image converted to binary")
    return image


def skeletonize_image(image: np.ndarray, method: str = "lee") -> np.ndarray:
    """Skeletonizes the binary image using the specified method."""
    logger.info(f"Skeletonizing image using method: {method}")
    skel = skeletonize(image // 255, method=method)
    skel = skel.astype(np.uint8)
    logger.info("Image skeletonization complete")
    return skel
