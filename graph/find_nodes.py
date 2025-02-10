from graph.node import Node
import numpy as np
from scipy.ndimage import convolve
import matplotlib.pyplot as plt


def find_nodes(skeleton: np.ndarray) -> dict[tuple[int, int], Node]:
    """Identifies node points (branch and edge) in the skeleton image."""
    print("[NODE_FINDING] Start finding nodes")

    kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)

    neighbor_degrees = convolve(skeleton, kernel, mode="constant", cval=0)

    # a branching point will have more than 2 neighboring pixels
    branch_nodes_image = (skeleton == 1) & (neighbor_degrees > 2)
    branch_nodes_image = branch_nodes_image.astype(np.uint8)

    # an edge will have exactly one neighboring pixel
    edge_nodes_image = (skeleton == 1) & (neighbor_degrees == 1)
    edge_nodes_image = edge_nodes_image.astype(np.uint8)

    coord_to_nodes: dict[tuple[int, int], Node] = {}
    h, w = branch_nodes_image.shape
    for y in range(h):
        for x in range(w):
            if branch_nodes_image[y, x] == 1 or edge_nodes_image[y, x] == 1:
                node = Node(x, y, len(coord_to_nodes))
                coord_to_nodes[(x, y)] = node

    print("[NODE_FINDING] Found", len(coord_to_nodes), "nodes")

    return coord_to_nodes


def plot_skeleton_overlayed_result(nodes: list[Node], skeleton: np.ndarray) -> None:
    bp_skel = skeleton.copy()
    for node in nodes:
        bp_skel[node.y, node.x] += 1
    plt.imshow(bp_skel)
    plt.show()
