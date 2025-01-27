import numpy as np
from PIL import Image


def downsample_image(image: np.ndarray, size=256) -> np.ndarray:
    """
    Downsamples a 2D (grayscale) image to (size x size) using PIL.

    :param image: np.ndarray of shape (H, W)
    :param size: target width and height, e.g., 256
    :return: downsampled np.ndarray of shape (size, size)
    """

    # Convert to uint8 if needed
    if image.dtype != np.uint8:
        # You may need scaling or clipping if your values aren't in [0, 255]
        image = image.astype(np.uint8)

    pil_img = Image.fromarray(image)
    pil_img = pil_img.resize(
        (size, size), Image.BICUBIC
    )  # or Image.ANTIALIAS (equiv. to BICUBIC now)

    return np.array(pil_img)


def mse(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Computes the mean squared error between two images of the same shape.
    """

    diff = (img1.astype(np.float32) - img2.astype(np.float32)) ** 2
    return np.mean(diff)
