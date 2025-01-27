import numpy as np
import logging
import cv2


def read_binary_image(filename: str, logger: logging.Logger) -> np.ndarray:
    """Reads an image from the specified filename and converts it to a binary image."""

    logger.info(f"Reading image from {filename}")

    image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    if image is None:
        logger.error(f"Failed to read image from {filename}")
        raise FileNotFoundError(f"Image file '{filename}' not found.")

    _, image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    logger.info("Image converted to binary")

    return image
