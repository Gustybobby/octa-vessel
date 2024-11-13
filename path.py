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
        mirrored_path = "-".join(reversed(path))
        if len(path) > 1 and mirrored_path not in all_paths:
            all_paths.add("-".join(path))

        for nb in graph[pt]:
            if nb in visited:
                continue

            new_path = path.copy()
            new_path.append(nb)

            if len(new_path) > 2:
                c = int(new_path[-1])
                b0 = int(new_path[-2])
                a0 = int(new_path[-3])
                vbc = pair_label[b0][c]["vectors"][0]

                pidx = -2
                while pidx > -len(new_path):
                    b = int(new_path[pidx])
                    a = int(new_path[pidx - 1])

                    vba = pair_label[a][b]["vectors"][1]
                    if pair_label[a][b]["sm"] == False:
                        break
                    pidx -= 1
                # if no large segment (sm = True) is found
                else:
                    vba = pair_label[a0][b0]["vectors"][1]

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


def prune_similar_paths(
    unique_paths: set[str],
    pair_label: list[list[dict | None]],
    graph: dict[str, set[str]],
) -> set[str]:
    pref_suf_paths = _prune_prefix_suffix_sm_segments(unique_paths, pair_label)
    pruned_paths = _prune_swappable_bps(pref_suf_paths, pair_label, graph)

    return pruned_paths


def _prune_prefix_suffix_sm_segments(
    unique_paths: set[str], pair_label: list[list[dict | None]]
) -> set[str]:
    # prune paths that start or end with sm segment
    pruned_unique_paths = unique_paths.copy()
    for path in unique_paths:
        bp_list = path.split("-")

        l1, l2 = bp_list[-2], bp_list[-1]

        prefix_path = "-".join(bp_list[:-1])
        last_is_ext_sm = pair_label[int(l1)][int(l2)]["sm"]
        if prefix_path in unique_paths and last_is_ext_sm:
            pruned_unique_paths.remove(path)
            continue

        l1, l2 = bp_list[0], bp_list[1]

        suffix_path = "-".join(bp_list[1:])
        first_is_sm = pair_label[int(l1)][int(l2)]["sm"]
        if suffix_path in unique_paths and first_is_sm:
            pruned_unique_paths.remove(path)

    logging.info(
        "after prefix-suffix pruning: "
        + str(len(pruned_unique_paths))
        + " unique paths"
    )
    return pruned_unique_paths


def _prune_swappable_bps(
    unique_paths: set[str],
    pair_label: list[list[dict | None]],
    graph: dict[str, set[str]],
) -> set[str]:
    pruned_paths = unique_paths.copy()
    for path in unique_paths:
        bp_list = path.split("-")
        for i in range(1, len(bp_list)):
            bp1, bp2 = bp_list[i - 1], bp_list[i]

            bp2_nbs = graph[bp2]
            bp1_nbs = graph[bp1]

            for nb in bp2_nbs:
                if nb == bp1:
                    continue
                # nb is a neighbor of bp2 and is a sm neighbor of bp1
                if nb in bp1_nbs and pair_label[int(nb)][int(bp1)]["sm"]:
                    sim_bp_list = bp_list.copy()
                    sim_bp_list[i - 1] = nb

                    sim_path = "-".join(sim_bp_list)
                    if sim_path in pruned_paths:
                        pruned_paths.remove(sim_path)

            for nb in bp1_nbs:
                if nb == bp2:
                    continue
                # nb is a neighbor of bp1 and is a sm neighbor of bp2
                if nb in bp2_nbs and pair_label[int(nb)][int(bp2)]["sm"]:
                    sim_bp_list = bp_list.copy()
                    sim_bp_list[i] = nb

                    sim_path = "-".join(sim_bp_list)
                    if sim_path in pruned_paths:
                        pruned_paths.remove(sim_path)

    logging.info(
        "after swappable branch points pruning: "
        + str(len(pruned_paths))
        + " unique paths"
    )

    return pruned_paths
