def highlight_branch_points(skeleton_image, beps: list[tuple[int, int]]):
    highlight_image = skeleton_image.copy() * 127
    for x, y in beps:
        highlight_image[y, x] = 255

    return highlight_image


def pretty_print(dict: dict):
    for k in dict:
        print(k, ": ", dict[k])
