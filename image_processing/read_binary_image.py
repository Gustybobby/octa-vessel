import numpy as np
import cv2


def read_binary_image(filename: str) -> np.ndarray:
    """Reads an image from the specified filename and converts it to a binary image."""

    print(f"[READ_BINARY] Reading image from {filename}")

    image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Image file '{filename}' not found.")

    _, image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    print("[READ_BINARY] Image converted to binary")

    return image
