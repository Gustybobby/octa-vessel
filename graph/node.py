from typing import Self


class Node:
    x: int
    y: int
    neighbors: dict[str, Self]

    def __init__(self, x: int, y: int, id: str) -> None:
        self.x = x
        self.y = y
        self.id = str(id)
        self.neighbors = {}

    def get_neighbors(self) -> list[Self]:
        return list(self.neighbors.values())

    def add_neighbor(self, node: Self) -> None:
        self.neighbors[node.id] = node

    def get_segment_key(self, nbr: Self) -> tuple[str, str]:
        return (self.id, nbr.id) if int(self.id) < int(nbr.id) else (nbr.id, self.id)
