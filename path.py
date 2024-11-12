import logging
import vector
from tqdm import trange

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)


def find_unique_paths(
    graph: dict[str, set[str]], pair_label: list[list[dict | None]], norm_cutoff: float
):
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

                if abs(vector.norm_dot(vbc, vba)) < norm_cutoff:
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


def prune_similar_paths(unique_paths: set[str], pair_label: list[list[dict | None]]):
    pruned_unique_paths = unique_paths.copy()
    for path in unique_paths:
        pt_list = path.split("-")

        l1, l2 = pt_list[-2], pt_list[-1]

        prefix_path = "-".join(pt_list[:-1])
        last_is_ext_or_sm = (
            "ext" in pair_label[int(l1)][int(l2)] or pair_label[int(l1)][int(l2)]["sm"]
        )
        if prefix_path in unique_paths and last_is_ext_or_sm:
            pruned_unique_paths.remove(path)
            continue

        l1, l2 = pt_list[0], pt_list[1]

        suffix_path = "-".join(pt_list[1:])
        first_is_ext_or_sm = (
            "ext" in pair_label[int(l1)][int(l2)] or pair_label[int(l1)][int(l2)]["sm"]
        )
        if suffix_path in unique_paths and first_is_ext_or_sm:
            pruned_unique_paths.remove(path)

    logging.info(
        "after similarity pruning: " + str(len(pruned_unique_paths)) + " unique paths"
    )
    return pruned_unique_paths
