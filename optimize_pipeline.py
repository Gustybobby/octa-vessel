from skimage.morphology import skeletonize
from scipy.ndimage import convolve
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import trange
from vector import find_end_vectors, norm_dot
from formatter import highlight_branch_points
import numpy as np
import cv2
import logging

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)


def read_binary_image(filename: str):
    image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    _, image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    logging.info("loaded " + filename)
    return image


def skeletonize_image(image, method="lee"):
    skel = skeletonize(image // 255, method=method)
    skel = skel.astype(np.uint8)

    logging.info("skeletonized image")
    return skel


def find_branch_points(skeleton_image) -> tuple[list[tuple[int, int]], dict]:
    kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)
    neighbors = convolve(skeleton_image, kernel, mode="constant", cval=0)

    bp_image = (skeleton_image == 1) & (neighbors > 2)
    bp_image = bp_image.astype(np.uint8)

    edge_image = (skeleton_image == 1) & (neighbors == 1)
    edge_image = edge_image.astype(np.uint8)

    beps = []
    h, w = bp_image.shape
    for y in range(h):
        for x in range(w):
            if bp_image[y, x] == 1 or edge_image[y, x] == 1:
                beps.append((x, y))

    cval_to_label = {}
    for i, (x, y) in enumerate(beps):
        coord_value = x * h + y
        cval_to_label[coord_value] = {"label": str(i), "coord": (x, y)}

    logging.info("found " + str(len(beps)) + " branch edge points")
    return beps, cval_to_label


def find_branch_segments(skeleton_image, beps: list[tuple[int, int]]):
    logging.info("start labeling connected components (segments)")
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
    branch_segments: list, cval_to_label: dict, sm_threshold: int
):
    logging.info("finding segment branch edge points")
    cnn_beps_list = []
    sm_list = []
    for i in trange(len(branch_segments)):
        segment = branch_segments[i]
        h, w = segment.shape

        non_zeros = cv2.findNonZero(segment)
        stack = [non_zeros[0][0]]
        visited = set()
        cnn_beps = set()
        while stack:
            x_curr, y_curr = stack.pop()

            if x_curr < 0 or y_curr < 0 or x_curr >= w or y_curr >= h:
                continue

            coord_value_current = x_curr * h + y_curr

            if coord_value_current in visited:
                continue
            visited.add(coord_value_current)

            if coord_value_current in cval_to_label:
                cnn_beps.add(cval_to_label[coord_value_current]["label"])
                continue

            if segment[y_curr, x_curr] == 0:
                continue

            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dx == dy == 0:
                        continue
                    stack.append((x_curr + dx, y_curr + dy))

        cnn_beps_list.append(tuple(sorted(list(cnn_beps), key=int)))
        sm_list.append(len(non_zeros) + 2 < sm_threshold)
        if len(cnn_beps_list[-1]) != 2:
            print(cnn_beps_list[-1])
            raise Exception("something is wrong")

    num_bps = len(cval_to_label)
    pair_label: list[list[dict | None]] = [
        [None for _ in range(num_bps)] for _ in range(num_bps)
    ]
    for i, (l1, l2) in enumerate(cnn_beps_list):
        pair_label[int(l1)][int(l2)] = {"label": i, "sm": sm_list[i]}
        pair_label[int(l2)][int(l1)] = {"label": i, "sm": sm_list[i]}
    return pair_label


def extend_connected_branch_points(
    branch_segments: list,
    pair_label: list[list[dict | None]],
    beps: list[tuple[int, int]],
    cval_to_label: dict,
    shape: tuple[int, int],
):
    logging.info("extending segments with connected branch points")
    h, _ = shape
    for i in trange(len(beps)):
        x, y = beps[i]
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == dy == 0:
                    continue
                coord_val = (x + dx) * h + (y + dy)
                if coord_val in cval_to_label:
                    nb = cval_to_label[coord_val]["label"]
                    l1, l2 = tuple(sorted([nb, str(i)], key=int))
                    if pair_label[int(l1)][int(l2)]:
                        continue
                    pair_label[int(l1)][int(l2)] = {
                        "label": len(branch_segments),
                        "sm": True,
                    }
                    pair_label[int(l2)][int(l1)] = {
                        "label": len(branch_segments),
                        "sm": True,
                    }

                    segment = np.zeros(shape, dtype=np.uint)
                    segment[y, x] = 255
                    x2, y2 = beps[int(nb)]
                    segment[y2, x2] = 255

                    branch_segments.append(segment)


def calc_pairs_end_vectors(branch_segments, beps, pair_label, depth: int):
    for l1 in range(len(pair_label)):
        for l2 in range(len(pair_label[l1])):
            if pair_label[l1][l2] is None or "vectors" in pair_label[l2][l1]:
                continue
            segment = branch_segments[pair_label[l1][l2]["label"]]
            pair_label[l1][l2]["vectors"] = find_end_vectors(
                segment, beps[l1], beps[l2], depth
            )
            pair_label[l2][l1]["vectors"] = tuple(
                reversed(pair_label[l1][l2]["vectors"])
            )


def construct_neighborhood_graph(
    pair_label: list[list[dict | None]],
) -> dict[str, set[str]]:
    graph = defaultdict(set)
    for l1 in range(len(pair_label)):
        for l2 in range(len(pair_label[l1])):
            if pair_label[l1][l2]:
                graph[str(l1)].add(str(l2))
                graph[str(l2)].add(str(l1))

    return graph


def find_unique_paths(graph: dict[str, set[str]], pair_label, norm_cutoff: float):
    all_paths: set[str] = set()
    visited: set[str] = set()

    def backtrack(pt: str, path: list[str]):
        if len(path) > 1 and "-".join(reversed(path)) not in all_paths:
            all_paths.add("-".join(path))
        for nb in graph[pt]:
            if nb in visited:
                continue

            new_path = path.copy()
            new_path.append(nb)

            if len(new_path) > 2:
                c = int(new_path[-1])
                b0 = int(new_path[-2])
                vbc = pair_label[b0][c]["vectors"][0]

                pidx = -2
                while pidx > -len(new_path):
                    b = int(new_path[pidx])
                    a = int(new_path[pidx - 1])

                    vba = pair_label[a][b]["vectors"][1]
                    if pair_label[a][b]["sm"] == False:
                        break
                    pidx -= 1
                else:
                    vba = pair_label[int(new_path[-3])][b0]["vectors"][1]

                if abs(norm_dot(vbc, vba)) < norm_cutoff:
                    continue

            visited.add(nb)

            backtrack(nb, new_path)

            visited.remove(nb)

    logging.info("finding unique paths")
    labels = sorted(list(graph), key=int)
    for i in trange(len(labels)):
        point = labels[i]
        visited.add(point)
        backtrack(point, [point])
        visited.remove(point)

    logging.info("found " + str(len(all_paths)) + " unique paths")
    return all_paths


if __name__ == "__main__":
    binary_image = read_binary_image("test_images\processed_pdr (100)_0.jpg")

    skeleton = skeletonize_image(binary_image, "lee")

    branching_points, cval_label_dict = find_branch_points(skeleton)

    segments = find_branch_segments(skeleton, branching_points)

    pair_label_arr = find_segment_pair_labels(segments, cval_label_dict, sm_threshold=5)
    extend_connected_branch_points(
        segments, pair_label_arr, branching_points, cval_label_dict, skeleton.shape
    )

    calc_pairs_end_vectors(segments, branching_points, pair_label_arr, depth=10)

    neighbor_graph = construct_neighborhood_graph(pair_label_arr)

    # cutoff higher = more_smooth_path
    upaths = find_unique_paths(neighbor_graph, pair_label_arr, norm_cutoff=0.707)

    hbeps_image = highlight_branch_points(skeleton, branching_points)

    raise "end"

    for path in upaths:
        p_list = path.split("-")
        img = np.zeros(skeleton.shape)
        for i in range(1, len(p_list)):
            bp1, bp2 = int(p_list[i - 1]), int(p_list[i])
            # print(bp1, bp2, pair_label_arr[bp1][bp2])
            segment_idx = pair_label_arr[bp1][bp2]["label"]
            img += segments[segment_idx]
            x1, y1 = branching_points[bp1]
            x2, y2 = branching_points[bp2]
            img[y1, x1] = 255
            img[y2, x2] = 255
        img = np.minimum(img, np.ones(skeleton.shape) * 255)
        # print(path)
        plt.imsave("res/" + path + ".png", img + hbeps_image * 0.8)

    plt.imshow(hbeps_image)
    plt.show()
