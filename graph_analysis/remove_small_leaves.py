import logging
from utils import edge


def remove_small_leaves(
    edge_lengths: dict[str, int],
    graph: dict[str, set],
    leaves_threshold: int,
    logger: logging.Logger,
):
    nodes_to_remove = set()
    for node in graph:
        if len(graph[node]) > 1:
            continue
        nbr = list(graph[node])[0]
        if edge.get_edge_length(node, nbr, edge_lengths) < leaves_threshold:
            nodes_to_remove.add(node)
            edge_key = edge.get_edge_key(node, nbr)
            if edge_key in edge_lengths:
                edge_lengths.pop(edge.get_edge_key(node, nbr))

    graph_nodes = list(graph.keys())
    for node in graph_nodes:
        if node not in graph:
            continue
        if node in nodes_to_remove:
            graph.pop(node)
        else:
            graph[node] = graph[node] - nodes_to_remove
        if len(graph[node]) == 0:
            graph.pop(node)

    logger.info("Total small leaves removed: " + str(len(nodes_to_remove)))

    return len(nodes_to_remove)
