from skimage import io
from skimage.morphology import skeletonize
from scipy.ndimage import convolve
from collections import defaultdict
import numpy as np
import cv2


def read_binary_image(filename: str):
    image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    _, image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    return image


def skeletonize_image(image, method="lee"):
    skel = skeletonize(image // 255, method=method)
    skel = skel.astype(np.uint8)

    return skel


def find_branch_edge_points(skeleton_image) -> list[tuple[int, int]]:
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

    return beps


def highlight_branch_edge_points(skeleton_image, beps: list[tuple[int, int]]):
    highlight_image = skeleton_image.copy()
    for x, y in beps:
        highlight_image[y, x] = 2

    return highlight_image


def find_neighborhood_graph(skeleton_image, beps: list[tuple[int, int]]):
    h, w = skeleton_image.shape

    # create a map: coord_value -> { label, coord }
    bp_info_map = {}
    for i, (x, y) in enumerate(beps):
        coord_value = x * h + y
        bp_info_map[coord_value] = {"label": str(i), "coord": (x, y)}

    # create a coord_value set
    bp_coord_value_set = set([x * h + y for (x, y) in beps])

    graph = defaultdict(set)
    for x, y in beps:
        visited = set()
        stack = [(x, y)]

        coord_value_start = x * h + y
        start_node: str = bp_info_map[coord_value_start]["label"]

        while stack:
            x_curr, y_curr = stack.pop()

            if x_curr < 0 or y_curr < 0 or x_curr >= w or y_curr >= h:
                continue

            coord_value_current = x_curr * h + y_curr
            not_path = skeleton_image[y_curr, x_curr] == 0

            if not_path or coord_value_current in visited:
                continue
            visited.add(coord_value_current)

            not_start = x_curr != x or y_curr != y
            is_bpe = coord_value_current in bp_coord_value_set

            if not_start and is_bpe:
                neighbor_node: str = bp_info_map[coord_value_current]["label"]
                graph[start_node].add(neighbor_node)
                continue

            # add all neighbors to stack
            stack.append((x_curr + 1, y_curr + 1))
            stack.append((x_curr + 1, y_curr))
            stack.append((x_curr + 1, y_curr - 1))
            stack.append((x_curr, y_curr + 1))
            stack.append((x_curr, y_curr - 1))
            stack.append((x_curr - 1, y_curr + 1))
            stack.append((x_curr - 1, y_curr))
            stack.append((x_curr - 1, y_curr - 1))

    return graph, bp_info_map


def pretty_print(dict: dict):
    for k in dict:
        print(k, ": ", dict[k])


if __name__ == "__main__":
    binary_image = read_binary_image("test_images/processed_pdr (98)_0.jpg")

    skeleton = skeletonize_image(binary_image, "lee")

    branching_points = find_branch_edge_points(skeleton)

    neighbor_graph, info_dict = find_neighborhood_graph(skeleton, branching_points)
    pretty_print(neighbor_graph)
    pretty_print(info_dict)

    hbeps_image = highlight_branch_edge_points(skeleton, branching_points)
    io.imshow(hbeps_image)
    io.show()

    # todo

    # find all branch segments and what pair (bp1, bp2/e_) they belong to (connected components)
    # i.e. 4 branches (A,B), (A,C), (A,e0), (A,e1) where e_ denotes an edge

    # traverse the neighborhood graph and get all possible connected permutations
    # i.e. { A: (B,C) } -> A,B and A,C

    # merge the branch segments to get image for all permutations (prune invalid branches via corner detection?)
