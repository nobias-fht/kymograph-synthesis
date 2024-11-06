from typing import Optional
from functools import lru_cache

import numpy as np
from numpy.typing import NDArray
import opensimplex
import microsim.schema as ms
from microsim._data_array import xrDataArray


class SimplexNoise:

    def __init__(
        self,
        scales: list[float],
        scale_weights: Optional[list[float]],
        max_intensity: float,
        seed: Optional[int] = None,
    ):
        if seed is not None:
            self.seed = seed
        else:
            opensimplex.random_seed()
            self.seed = opensimplex.get_seed()

        self._max_intensity = max_intensity
        self._scales = scales
        self._scale_weights = scale_weights

    def render(self, space: xrDataArray, xp: ms.NumpyAPI | None = None) -> xrDataArray:
        space += self.noise_array(dims=space.shape)
        return space

    @lru_cache
    def noise_array(self, dims: tuple[int, int, int]) -> NDArray:
        opensimplex.seed(self.seed)
        noise_array = np.zeros(dims)
        for scale, weight in zip(self._scales, self._scale_weights):
            noise_grid = [np.arange(dim) * scale / max(dims) for dim in reversed(dims)]
            noise_array += weight * opensimplex.noise3array(*noise_grid)

        noise_array = (noise_array / noise_array.max()) * self._max_intensity
        # clip < 0
        noise_array[noise_array < 0] = 0
        return noise_array