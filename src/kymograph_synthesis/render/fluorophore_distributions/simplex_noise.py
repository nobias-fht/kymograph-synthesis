from functools import lru_cache
from typing import cast

import numpy as np
from numpy.typing import NDArray
import opensimplex
import microsim.schema as ms
from microsim._data_array import xrDataArray


class SimplexNoise:

    def __init__(
        self,
        noise_scales: list[float],
        scale_weights: list[float],
        max_fluorophore_count: float,
        seed: int,
    ):
        self._seed = seed
        self._noise_scales = noise_scales
        self._scale_weights = scale_weights
        self._max_intensity = max_fluorophore_count

    def render(self, space: xrDataArray, xp: ms.NumpyAPI | None = None) -> xrDataArray:
        truth_space = cast(ms.space.Space, space.attrs["space"])
        space += self.noise_array(dims=space.shape, scale=truth_space.scale)
        return space

    @lru_cache
    def noise_array(
        self, dims: tuple[int, int, int], scale: tuple[float, float, float]
    ) -> NDArray:
        opensimplex.seed(self._seed)
        noise_array = np.zeros(dims)
        for noise_scale, weight in zip(self._noise_scales, self._scale_weights):
            noise_grid = [
                np.linspace(0, noise_scale * s * dim, dim)
                for s, dim in zip(scale, dims)
            ]
            noise_array += weight * opensimplex.noise3array(*reversed(noise_grid))

        noise_array = (noise_array / noise_array.max()) * self._max_intensity
        # clip < 0
        noise_array[noise_array < 0] = 0
        return noise_array