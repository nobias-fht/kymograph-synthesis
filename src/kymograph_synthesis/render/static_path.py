from typing import Protocol

import numpy as np
from numpy.typing import NDArray, ArrayLike


class StaticPath(Protocol):

    def __call__(self, ratio: NDArray) -> NDArray: ...


class LinearPath:

    def __init__(self, start: NDArray, end: NDArray):
        self.dims = len(start)
        self.start = start
        self.end = end
        self.direction = end - start

    def __call__(self, ratio: NDArray) -> NDArray:
        shape = ratio.shape
        result = self.start.reshape(-1, 1) + np.outer(self.direction, ratio.flatten())
        return result.reshape(self.dims, *shape)
