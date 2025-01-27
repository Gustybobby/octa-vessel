import numpy as np
import logging
from tqdm import tqdm
from utils.edge import get_edge_key
from image_processing.image_similarity import downsample_image, mse


def build_distinct_images(
    valid_path_list: list[tuple[str]],
    vessel_segments: dict[str, np.ndarray],
    mse_threshold: float,
    logger: logging.Logger,
):
    valid_paths = {}
    downsampled_images = {}
    for path_tuple in tqdm(valid_path_list, desc="Building images"):
        segments = [
            vessel_segments[get_edge_key(path_tuple[i], path_tuple[i + 1])].astype(
                np.uint8
            )
            for i in range(len(path_tuple) - 1)
        ]
        image = np.maximum.reduce(segments)
        downsampled = downsample_image(image, 256)

        if _is_image_similar(downsampled, downsampled_images, mse_threshold):
            continue

        valid_paths[path_tuple] = image
        downsampled_images[path_tuple] = downsampled

    logger.info(f"Total distinct images: {len(valid_paths)}")

    return valid_paths


def _is_image_similar(
    downsampled: np.ndarray,
    downsampled_images: dict[str, np.ndarray],
    mse_threshold: float,
) -> bool:
    # check with existing distinct path tuples
    for downsampled_path_tuple in downsampled_images:
        # compare downsampled candidate with existing downsampled
        error = mse(downsampled_images[downsampled_path_tuple], downsampled)
        if error <= mse_threshold:
            return True

    return False
