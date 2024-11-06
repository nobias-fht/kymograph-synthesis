import numpy as np
from numpy.typing import NDArray
import microsim.schema as ms
from microsim._data_array import xrDataArray

from ..static_path import StaticPath


class ParticleSystem:

    def __init__(self, coords: NDArray, intensities: NDArray):

        # [0, 1] in coordinate space will be scaled to the extent of each dimension
        # TODO: make sure dimensions match
        self.coords = coords  # N x D
        self.intensities = intensities  # N

        self.n_particles = self.coords.shape[1]

    @classmethod
    def on_static_path(
        cls, static_path: StaticPath, path_positions: NDArray, intensities: NDArray
    ):
        space_coords = static_path(path_positions)
        return cls(coords=space_coords, intensities=intensities)

    def render(self, space: xrDataArray, xp: ms.NumpyAPI | None = None) -> xrDataArray:
        # map coord space to render space
        # unit_distance = max(space.shape)
        # space_coords = self.coords * unit_distance

        space_coords = self.coords * np.array(space.shape).reshape(1, -1)
        indices = np.round(space_coords).astype(int)

        # remove out of bounds indices
        out_of_bounds = np.logical_or(
            (indices < 0).any(axis=1),
            (indices >= np.array(space.shape).reshape(1, -1)).any(axis=1),
        )
        indices = indices[~out_of_bounds]

        space[indices[:, 0], indices[:, 1], indices[:, 2]] += self.intensities[
            ~out_of_bounds
        ]
        return space
