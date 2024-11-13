class BranchPointData:
    label: str
    coord: tuple[int, int]

    def __init__(self, label: str, coord: tuple[int, int]):
        self.label = label
        self.coord = coord

    def getX(self) -> int:
        return self.coord[0]

    def getY(self) -> int:
        return self.coord[1]
