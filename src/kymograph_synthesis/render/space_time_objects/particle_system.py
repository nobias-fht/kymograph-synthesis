import numpy as np
from numpy.typing import NDArray

from ..static_path import StaticPath


class ParticleSystem:

    def __init__(self, coords: NDArray, intensities: NDArray):

        # TODO: make sure dimensions match
        self.coords = coords  # T x N x D
        self.intensities = intensities  # T x N

        self.n_particles = self.coords.shape[1]

    @classmethod
    def on_static_path(
        cls, static_path: StaticPath, path_positions: NDArray, intensities: NDArray
    ):
        space_coords = static_path(path_positions)
        return cls(coords=space_coords, intensities=intensities)

    def render(self, t: int, space: NDArray) -> NDArray:
        for particle_idx in range(self.n_particles):
            particle_coords = self.coords[t, particle_idx]
            indices = np.round(particle_coords).astype(int)
            # skip if out of space bounds
            if (indices < 0).any():
                continue
            if (indices >= np.array(space.shape)).any():
                continue
            space[*indices] += self.intensities[t, particle_idx]
        return space
