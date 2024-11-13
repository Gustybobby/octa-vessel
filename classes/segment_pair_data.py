class SegmentPairData:
    label: int
    sm: bool
    ext: bool
    count: int
    vectors: tuple[tuple[int, int], tuple[int, int]] | None

    def __init__(self, label: int, sm: bool, ext: bool, count: int):
        self.label = label
        self.sm = sm
        self.ext = ext
        self.count = count
        self.vectors = None

    def copy(self):
        return SegmentPairData(self.label, self.sm, self.ext, self.count)
