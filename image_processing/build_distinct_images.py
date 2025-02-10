import numpy as np
from tqdm import tqdm
from graph.extract_segments import get_segment_key
from image_processing.image_similarity import downsample_image, mse


def build_distinct_images(
    valid_paths: set[tuple[str]],
    segments: dict[tuple[str, str], set[tuple[int, int]]],
    mse_threshold: float,
    shape: tuple[int, int],
):
    segment_images = {}
    for key, pixels in segments.items():
        image = np.zeros(shape)
        for x, y in pixels:
            image[y, x] = 255
        segment_images[key] = image

    valid_paths_images = {}
    downsampled_images = {}

    for path in tqdm(
        sorted(list(valid_paths), key=lambda x: (len(x), x)), desc="Building images"
    ):
        start = 0
        cached_image = np.zeros(shape)
        for i in range(1, len(path)):
            sub_path = tuple(path[: i + 1])
            if sub_path in valid_paths_images:
                cached_image = valid_paths_images[sub_path].copy()
                start = i

        image = np.maximum.reduce(
            [
                segment_images[get_segment_key(path[i], path[i + 1])].astype(np.uint8)
                for i in range(start, len(path) - 1)
            ]
        )
        image = np.maximum(image, cached_image)

        downsampled = downsample_image(image, 256)

        if _is_image_similar(downsampled, downsampled_images, mse_threshold):
            continue

        valid_paths_images[path] = image
        downsampled_images[path] = downsampled

    print(f"Total distinct images: {len(valid_paths_images)}")

    return valid_paths_images


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
