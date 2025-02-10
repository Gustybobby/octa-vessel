from graph.node import Node
from graph.extract_segments import get_segment_key
from tqdm import tqdm
import math


def build_valid_paths(
    coord_to_nodes: dict[tuple[int, int], Node],
    segments: dict[tuple[str, str], set[tuple[int, int]]],
    small_segment_threshold: int,
    num_points: int,
    angle_threshold: int,
    min_final_length: int,
    max_depth: int,
) -> list[tuple[str]]:
    def backtrack(node: Node, unchecked_junc_idx: int):
        exist = tuple(path) in valid_paths
        reversed_exist = tuple(reversed(path)) in valid_paths
        small_final = get_path_length(path, segments) < min_final_length
        last_is_small = (
            len(path) > 1
            and get_path_length(path[-2:], segments) < small_segment_threshold
        )
        if not exist and not reversed_exist and not small_final and not last_is_small:
            valid_paths.add(tuple(path))

        if len(path) >= max_depth:
            return

        for nbr in sorted(node.get_neighbors(), key=lambda n: int(n.id)):
            if nbr.id in visited:
                continue

            path.append(nbr.id)

            small_start = (
                len(path) == 2
                and get_path_length(path, segments) < small_segment_threshold
            )
            # check if the path doesn't start with a small segment
            if not small_start:
                new_unchecked_idx = unchecked_junc_idx
                angle = None
                # if path junction can be checked
                if len(path) > 2:
                    # checked along the path if the path is smooth (where it is checkable (suffix length >= num points))
                    while new_unchecked_idx < len(path) - 1:
                        junc_angle = _junc_angle(
                            path, id_to_nodes, segments, new_unchecked_idx, num_points
                        )
                        # reached a point where it cannot be checked any more, break
                        if junc_angle is None:
                            break

                        # track minimum angle and early stop when < threshold
                        angle = junc_angle if angle is None else min(junc_angle, angle)
                        if angle < angle_threshold:
                            break

                        # update the unchecked junction
                        new_unchecked_idx += 1
                else:
                    # just move unchecked junction to the end
                    new_unchecked_idx = len(path) - 1

                if angle is None or angle > angle_threshold:
                    visited.add(nbr.id)
                    backtrack(nbr, new_unchecked_idx)
                    visited.remove(nbr.id)

            path.pop()

    id_to_nodes = {}
    for node in coord_to_nodes.values():
        id_to_nodes[node.id] = node

    valid_paths: set[tuple[str]] = set()

    for node in tqdm(coord_to_nodes.values(), desc="backtracking paths"):
        path: list[str] = [node.id]
        visited: set[str] = set([node.id])
        backtrack(node, 0)

    print(f"[PATH_FINDING] Found {len(valid_paths)} valid paths")
    return sorted(list(valid_paths), key=lambda x: (len(x), x))


def get_path_length(
    path: list[str], segments: dict[tuple[str, str], set[tuple[int, int]]]
) -> int:
    return sum(
        [
            len(segments[get_segment_key(path[i - 1], path[i])])
            for i in range(1, len(path))
        ]
    )


def _junc_angle(
    path: list[str],
    id_to_nodes: dict[str, Node],
    segments: dict[tuple[str, str], set[tuple[int, int]]],
    junc_idx: int,
    num_points: int,
) -> float | None:
    def dfs(
        curr: tuple[int, int], count: int, pixels: set[tuple[int, int]]
    ) -> tuple[int, int]:
        if count == num_points:
            return curr
        x, y = curr

        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue

                next_coord = (x + dx, y + dy)

                if next_coord in visited:
                    continue

                if next_coord in pixels:
                    visited.add(next_coord)
                    return dfs(next_coord, count + 1, pixels)

        return curr

    #       junc
    # prefix  |  suffix
    #   0, 1, 2, 3, 4, 5

    junc_node = id_to_nodes[path[junc_idx]]
    junc_coord = (junc_node.x, junc_node.y)

    suffix_length = get_path_length(path[junc_idx:], segments)
    if suffix_length < num_points:
        return None

    suffix_pixels = set()
    for i in range(junc_idx, len(path) - 1):
        suffix_pixels = suffix_pixels.union(
            segments[get_segment_key(path[i], path[i + 1])]
        )
        if len(suffix_pixels) >= num_points:
            break
    visited = set([junc_coord])
    suffix_end = dfs(junc_coord, 1, suffix_pixels)
    if suffix_end == junc_coord:
        print(path, junc_idx, suffix_pixels, visited)
        raise Exception("no suffix")

    prefix_pixels = set()
    for i in range(junc_idx, 0, -1):
        prefix_pixels = prefix_pixels.union(
            segments[get_segment_key(path[i], path[i - 1])]
        )
        if len(prefix_pixels) >= num_points:
            break
    visited = set([junc_coord])
    prefix_end = dfs(junc_coord, 1, prefix_pixels)
    if prefix_end == junc_coord:
        print(path, junc_idx, prefix_pixels, visited)
        raise Exception("no prefix")

    suffix_vector = _get_vector(suffix_end, junc_coord)
    prefix_vector = _get_vector(prefix_end, junc_coord)

    return _get_vectors_inner_angle(prefix_vector, suffix_vector)


def _get_vector(head: tuple[int, int], tail: tuple[int, int]) -> tuple[int, int]:
    return (head[0] - tail[0], head[1] - tail[1])


def _get_vectors_inner_angle(v1: tuple[int, int], v2: tuple[int, int]) -> float:
    inner_product = v1[0] * v2[0] + v1[1] * v2[1]
    mag1 = math.sqrt(v1[0] ** 2 + v1[1] ** 2)
    mag2 = math.sqrt(v2[0] ** 2 + v2[1] ** 2)
    cosine = inner_product / (mag1 * mag2)
    return math.acos(min(max(-1, cosine), 1)) * 180 / math.pi
