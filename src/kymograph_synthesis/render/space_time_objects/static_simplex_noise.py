from typing import Optional

import numpy as np
from numpy.typing import NDArray
import opensimplex


class StaticSimplexNoise:

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

        self.max_intensity = max_intensity
        self._scales = scales
        self._scale_weights = scale_weights

        self._previous_dims: Optional[NDArray[np.int_]] = None
        self.noise_array: Optional[NDArray] = None

    def render(self, t: int, space: NDArray):
        dims = np.array(space.shape)
        noise_array = self.get_noise(dims)
        space += noise_array * self.max_intensity
        return space
    
    def get_noise(self, dims: NDArray[np.int_]) -> NDArray:
        if not (dims == self._previous_dims).all():
            noise_array = self._generate_noise(dims=dims)
            self._previous_dims = dims
            self.noise_array = noise_array
        return self.noise_array

    def _generate_noise(self, dims: NDArray[np.int_]) -> NDArray:
        opensimplex.seed(self.seed)
        noise_array = np.zeros(dims)
        for scale, weight in zip(self._scales, self._scale_weights):
            noise_grid = [np.arange(dim) * scale / dims.max() for dim in reversed(dims)]
            noise_array += weight * opensimplex.noise3array(*noise_grid)
        noise_array = (noise_array - noise_array.min()) / (
            noise_array.max() - noise_array.min()
        )
        return noise_array

