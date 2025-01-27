import utilities


def get_edge_key(a: str, b: str) -> str:
    return f"{min(a, b)} - {max(a, b)}"


def get_edge_length(a: str, b: str, edge_lengths: dict[str, int]) -> int:
    return edge_lengths.get(get_edge_key(a, b), 0)


def is_small_segment(
    u: str, v: str, threshold: int, edge_lengths: dict[str, int]
) -> bool:
    return get_edge_length(u, v, edge_lengths) <= threshold


def fit_angle_between_two_edges(
    edge1: str, edge2: str, points_dict: dict, vessel_segments
) -> float:
    return utilities.fit_tangents_at_junction(
        points_dict, edge1, edge2, vessel_segments
    )
