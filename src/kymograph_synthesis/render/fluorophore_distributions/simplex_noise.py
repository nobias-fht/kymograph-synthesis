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
        noise_scales: tuple[float, ...],
        scale_weights: tuple[float, ...],
        max_fluorophore_count_per_nm3: float,
        seed: int,
    ):
        self._seed = seed
        self._noise_scales = noise_scales
        self._scale_weights = scale_weights
        # max fluorphore_count per nm
        self._max_fluorophore_count_per_nm3 = max_fluorophore_count_per_nm3

    def render(self, space: xrDataArray, xp: ms.NumpyAPI | None = None) -> xrDataArray:
        truth_space = cast(ms.space.Space, space.attrs["space"])
        space += self.noise_array(dims=space.shape, scale=truth_space.scale)
        return space



    def noise_array(
        self, dims: tuple[int, int, int], scale: tuple[float, float, float]
    ) -> NDArray:
        return _noise_array(
            dims=dims, 
            scale=scale, 
            seed=self._seed, 
            noise_scales=self._noise_scales,
            scale_weights=self._scale_weights, 
            max_fluorophore_count_per_nm3=self._max_fluorophore_count_per_nm3
        )


@lru_cache(maxsize=1)
def _noise_array(
    dims: tuple[int, int, int], 
    scale: tuple[float, float, float], 
    seed: int, 
    noise_scales: tuple[float, ...], 
    scale_weights: tuple[float, ...], 
    max_fluorophore_count_per_nm3: float
) -> NDArray:
    opensimplex.seed(seed)
    noise_array = np.zeros(dims)
    for noise_scale, weight in zip(noise_scales, scale_weights):
        noise_grid = [
            np.linspace(0, noise_scale * s * dim, dim)
            for s, dim in zip(scale, dims)
        ]
        noise_array += weight * opensimplex.noise3array(*reversed(noise_grid))

    noise_array = (
        (noise_array / noise_array.max())
        * (max_fluorophore_count_per_nm3 * 1e6) # per um3
        * np.prod(scale)
    )
    # clip < 0
    noise_array[noise_array < 0] = 0
    return noise_array