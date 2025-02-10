from skimage.morphology import skeletonize
import numpy as np


def skeletonize_image(image: np.ndarray, method: str = "lee") -> np.ndarray:
    """Skeletonizes the binary image using the specified method. {lee,zhang}"""

    print(f"[SKELETONIZE] Skeletonizing image using method: {method}")

    skeleton = skeletonize(image // 255, method=method)
    skeleton = skeleton.astype(np.uint8)

    print("[SKELETONIZE] Image skeletonization complete")

    return skeleton
