import cv2
import numpy as np
import logging
import matplotlib.pyplot as plt
from tqdm import trange
from classes.branch_point_data import BranchPointData
from classes.segment_pair_data import SegmentPairData

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)


def find_branch_segments(skeleton_image, beps: list[tuple[int, int]]):
    beps_overlay = np.zeros(skeleton_image.shape, dtype=np.uint8)
    for x, y in beps:
        beps_overlay[y, x] = 255
    segmented_skeleton = cv2.subtract(skeleton_image, beps_overlay)

    num_labels, labels = cv2.connectedComponents(segmented_skeleton)
    segments = [
        (labels == label).astype(np.uint8) * 255 for label in range(1, num_labels)
    ]

    logging.info("finished labeling connected components (segments)")
    return segments


def find_segment_pair_labels(
    branch_segments: list[np.ndarray],
    cval_to_data: dict[int, BranchPointData],
    sm_threshold: int,
):
    logging.info("finding segment branch edge points")
    cnn_beps_list = []
    sm_list = []
    for i in trange(len(branch_segments)):
        segment = branch_segments[i]
        h, w = segment.shape
        assert isinstance(h, int), "h is not int"
        assert isinstance(w, int), "w is not int"

        non_zeros = cv2.findNonZero(segment)
        stack: list[tuple[int, int]] = [non_zeros[0][0]]
        visited: set[int] = set()
        cnn_beps: set[str] = set()
        while stack:
            x_curr, y_curr = stack.pop()

            if x_curr < 0 or y_curr < 0 or x_curr >= w or y_curr >= h:
                continue

            coord_value_current = x_curr * h + y_curr

            if coord_value_current in visited:
                continue
            visited.add(coord_value_current)

            if coord_value_current in cval_to_data:
                cnn_beps.add(cval_to_data[coord_value_current].label)
                continue

            if segment[y_curr, x_curr] == 0:
                continue

            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dx == dy == 0:
                        continue
                    stack.append((x_curr + dx, y_curr + dy))

        if len(cnn_beps) > 2:
            print(cnn_beps)
            raise Exception("something is wrong")
        elif len(cnn_beps) == 1:
            cnn_beps: list[str] = list(cnn_beps)
            cnn_beps.append(cnn_beps[0])

        pair = tuple(sorted(list(cnn_beps), key=int))
        cnn_beps_list.append(pair)
        sm_list.append(len(non_zeros) + 2 < sm_threshold)

    num_bps = len(cval_to_data)
    pair_data: list[list[SegmentPairData | None]] = [
        [None for _ in range(num_bps)] for _ in range(num_bps)
    ]
    for i, (l1, l2) in enumerate(cnn_beps_list):
        l1, l2 = int(l1), int(l2)
        pair_data[l1][l2] = SegmentPairData(
            label=i,
            sm=sm_list[i],
            ext=False,
            count=cv2.countNonZero(branch_segments[i]),
        )
        pair_data[l2][l1] = pair_data[l1][l2].copy()
    return pair_data


def extend_connected_branch_points(
    branch_segments: list,
    pair_data: list[list[SegmentPairData | None]],
    beps: list[tuple[int, int]],
    cval_to_data: dict[int, BranchPointData],
    shape: tuple[int, int],
):
    h, _ = shape
    for i in range(len(beps)):
        x, y = beps[i]
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == dy == 0:
                    continue
                coord_val = (x + dx) * h + (y + dy)
                if coord_val in cval_to_data:
                    nb = cval_to_data[coord_val].label
                    l1, l2 = tuple(sorted([nb, str(i)], key=int))
                    l1, l2 = int(l1), int(l2)

                    if pair_data[l1][l2]:
                        continue

                    pair_data[l1][l2] = {
                        "label": len(branch_segments),
                        "sm": True,
                        "ext": True,
                        "count": 2,
                    }
                    pair_data[l1][l2] = SegmentPairData(
                        label=len(branch_segments), sm=True, ext=True, count=2
                    )
                    pair_data[l2][l1] = pair_data[l1][l2].copy()

                    segment = np.zeros(shape, dtype=np.uint8)
                    segment[y, x] = 255
                    x2, y2 = beps[int(nb)]
                    segment[y2, x2] = 255

                    branch_segments.append(segment)


def segment_union(
    unique_paths: set[str],
    segments: list[np.ndarray],
    pair_data: list[list[SegmentPairData | None]],
    skeleton: np.ndarray,
    beps: list[tuple[int, int]],
    frac_length_cutoff: float,
    save_dir: str,
    save=False,
    overlay=False,
):
    logging.info("merging path segments")
    unique_paths: list[str] = sorted(list(unique_paths))
    count = 0
    last_path = None
    for i in trange(len(unique_paths)):
        u_path = unique_paths[i]
        p_list = u_path.split("-")

        ex_last = "-".join(p_list[:-1])
        if ex_last == last_path:
            start = len(p_list) - 1
        else:
            start = 1
            img = np.zeros(skeleton.shape, dtype=np.uint32)

        for j in range(start, len(p_list)):
            bp1, bp2 = int(p_list[j - 1]), int(p_list[j])

            segment = segments[pair_data[bp1][bp2].label]
            (x1, y1), (x2, y2) = beps[bp1], beps[bp2]
            segment[y1, x1] = segment[y2, x2] = 255
            img += segment
        img = np.minimum(
            img, np.ones(skeleton.shape, dtype=np.uint8) * 255, dtype=np.uint8
        )

        non_zero_count = cv2.countNonZero(img)
        if non_zero_count <= frac_length_cutoff * skeleton.shape[0]:
            continue
        count += 1
        last_path = u_path

        if save:
            save_path = save_dir + "/" + str(count) + ".png"
            if overlay:
                overlayed_image = img.astype(np.uint16) + (skeleton * 128)
                plt.imsave(save_path, overlayed_image)
            else:
                cv2.imwrite(save_path, img)
    if save:
        logging.info("saved " + str(count) + " paths")
    else:
        logging.info("found final " + str(count) + " paths")
