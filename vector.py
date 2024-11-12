import math


def find_end_vectors(
    segment, ep1: tuple[int, int], ep2: tuple[int, int], depth: int
) -> tuple[tuple[int, int], tuple[int, int]]:
    h, w = segment.shape
    vectors = []
    for pt in [ep1, ep2]:
        visited = set()
        stack = [pt]
        path = []
        while stack:
            x_curr, y_curr = stack.pop()

            if x_curr < 0 or y_curr < 0 or x_curr >= w or y_curr >= h:
                continue

            coord_value_current = x_curr * h + y_curr
            not_path = (
                segment[y_curr, x_curr] == 0
                and not (x_curr == ep1[0] and y_curr == ep1[1])
                and not (x_curr == ep2[0] and y_curr == ep2[1])
            )

            if not_path or coord_value_current in visited:
                continue
            visited.add(coord_value_current)

            path.append((x_curr, y_curr))

            if len(path) >= depth:
                break

            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dx == dy == 0:
                        continue
                    stack.append((x_curr + dx, y_curr + dy))

        vectors.append((path[-1][0] - path[0][0], path[-1][1] - path[0][1]))

    return tuple(vectors)


def norm_dot(v1: tuple[int, int], v2: tuple[int, int]) -> float:
    x1, y1 = v1
    x2, y2 = v2
    return (x1 * x2 + y1 * y2) / (math.sqrt(x1**2 + y1**2) * math.sqrt(x2**2 + y2**2))
