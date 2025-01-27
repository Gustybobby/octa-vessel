import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)


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


def remove_very_similar_images_with_downsample(
    image_dict: dict,
    mse_threshold: float = 1e-3,
    size: int = 256,
    show_duplicates: bool = True,
) -> dict:
    """
    Removes images that are very similar or identical, based on a downsampling step + MSE.

    :param image_dict: { key: ndarray }, each is a grayscale image of shape (H, W)
    :param mse_threshold: If MSE is <= this value, we treat images as duplicates
    :param size: Downsample size for width and height
    :param show_duplicates: Whether to display duplicates side by side (matplotlib)

    :return: A filtered dictionary { key: ndarray } with near-duplicates removed.
    """
    # Precompute downsampled versions to speed comparisons
    downsampled_dict = {}
    for key, img in image_dict.items():
        downsampled_dict[key] = downsample_image(img, size=size)

    unique_images = {}
    keys = list(image_dict.keys())
    keys.sort(key=lambda x: (len(x), x))

    for i, key_i in tqdm(
        enumerate(keys),
        desc=f"Removing Duplicates",
        unit="image",
        total=len(image_dict),
    ):
        isDupe = False
        # Compare it against all unprocessed images
        for j in range(i + 1, len(keys)):
            key_j = keys[j]

            # First check shape (at original resolution); skip if obviously different
            if image_dict[key_i].shape != image_dict[key_j].shape:
                continue

            # Compute MSE on downsampled images
            error = mse(downsampled_dict[key_i], downsampled_dict[key_j])

            # if i dupe with j: set isDupe flag to True and break
            if error <= mse_threshold:
                if show_duplicates:
                    fig, axes = plt.subplots(1, 2, figsize=(6, 3))

                    axes[0].imshow(image_dict[key_i], cmap="gray")
                    axes[0].set_title(f"Original: {key_i}")
                    axes[0].axis("off")

                    axes[1].imshow(image_dict[key_j], cmap="gray")
                    axes[1].set_title(f"Duplicate: {key_j}\nMSE={error:.6f}")
                    axes[1].axis("off")

                    plt.tight_layout()
                    plt.show()
                isDupe = True
                break

        if not isDupe:
            unique_images[key_i] = image_dict[key_i]

    logger.info(f"Total images removed as duplicates: {len(keys) - len(unique_images)}")
    return unique_images


# ------------------------------------------------------
# Example Usage:
# ------------------------------------------------------
# images_dict = {
#     "img1": ndarray1,  # shape (1024, 1024)
#     "img2": ndarray2,
#     ...
# }
#
# filtered = remove_very_similar_images_with_downsample(
#     images_dict,
#     mse_threshold=0.5,  # adjust this threshold as needed
#     size=256,
#     show_duplicates=True
# )
#
# This will:
#  1. Downsample each image to 256x256.
#  2. Compare MSE of downsampled images.
#  3. If MSE <= 0.5, consider them duplicates and show them side by side.
