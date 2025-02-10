# visualization.py

import os
import cv2
import numpy as np
import logging
from tqdm import tqdm
from typing import Dict
import csv
from config import CROP, MARGIN

logger = logging.getLogger(__name__)


def classify_tortuous(path, tortuosity_index, counts):
    if path not in tortuosity_index:
        raise Exception(f"TI for path {path} not calculated")
    ti = tortuosity_index[path]
    if ti < 1.3:
        counts[0] += 1
        return "non_tortuous"
    elif ti > 1.5:
        counts[1] += 1
        return "tortuous"
    else:
        counts[2] += 1
        return "unknown"


def crop_image(image, margin, log_file, filename, writer):
    """
    Crops the image to keep the region with only with max pixel intensity plus the margin on all sides
    It will also record the pixel coordinates in the original image where the vessel occurs (can be used to localize the vessel later on in the original image.)
    """
    # Ensure the image is a NumPy array
    if not isinstance(image, np.ndarray):
        raise TypeError("The input image must be a NumPy ndarray.")
    # Find the coordinates where pixel value is 255
    ys, xs = np.where(image == 255)
    # log_file.write(f"{filename}, {xs}, {ys} \n")
    writer.writerow([filename, xs.tolist(), ys.tolist()])
    if ys.size == 0 or xs.size == 0:
        raise ValueError("No pixels with value 255 found in the image.")

    # Determine the bounding box
    top, bottom = max(ys.min() - margin, 0), min(ys.max() + margin, image.shape[0])
    left, right = max(xs.min() - margin, 0), min(xs.max() + margin, image.shape[1])

    # Crop the image
    cropped_image = image[top : bottom + 1, left : right + 1]

    return cropped_image


def save_images(
    valid_paths: Dict[tuple[str, ...], np.ndarray],
    result_dir: str,
    OVERLAY_IMAGE,
    OVERLAY_INTENSITY,
    skeleton,
    tortuosity_index,
    key=None,
):
    """Saves the vessel segment images to the result directory without borders."""
    logger.info("Saving vessel segment images")

    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(os.path.join(result_dir, "tortuous"), exist_ok=True)
    os.makedirs(os.path.join(result_dir, "non_tortuous"), exist_ok=True)
    os.makedirs(os.path.join(result_dir, "unknown"), exist_ok=True)

    i = 0
    counts = [0, 0, 0]
    small_removed = 0

    sorted_path_keys = sorted(valid_paths.keys(), key=key)
    path_log_file = open(
        "\\".join(result_dir.split("\\")[:-1]) + "\\vessels_localized_log.csv",
        "a",
        newline="",
    )
    writer = csv.writer(path_log_file)

    for path in tqdm(
        sorted_path_keys, desc=f"Saving images to {result_dir}", unit="image"
    ):  # Construct a filename from the path tuple
        filename = str(i) + ".png"
        sub_dir = classify_tortuous(path, tortuosity_index, counts)

        filepath = os.path.join(result_dir, sub_dir, filename)

        if OVERLAY_IMAGE:
            overlayed_image = valid_paths[path].astype(np.uint16) + (
                skeleton * 128 * OVERLAY_INTENSITY
            )
            overlayed_image = np.clip(overlayed_image, 0, 255).astype(np.uint8)
            if CROP:
                cv2.imwrite(
                    filepath,
                    crop_image(
                        overlayed_image, MARGIN, path_log_file, filename, writer
                    ),
                )
            else:
                cv2.imwrite(filepath, overlayed_image)
        else:
            if CROP:
                cv2.imwrite(
                    filepath,
                    crop_image(
                        valid_paths[path], MARGIN, path_log_file, filename, writer
                    ),
                )
            else:
                cv2.imwrite(filepath, valid_paths[path])

        logger.debug(f"Saved image: {filepath}")
        i += 1
    logger.info(
        f"Tortuous : {counts[1]}, Non-tortuous : {counts[0]}, Unknown : {counts[2]}"
    )
    if small_removed > 0:
        logger.info("Total small single-level paths removed: " + str(small_removed))

    path_log_file.close()
