import numpy as np
import xarray as xr
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
        #TODO: check how non-equal spatial dimensions are treated in microsim
        space_coords = self.coords * max(space.shape)
        indices = np.round(space_coords).astype(int)

        # remove out of bounds indices
        out_of_bounds = np.logical_or(
            (indices < 0).any(axis=1),
            (indices >= np.array(space.shape).reshape(1, -1)).any(axis=1),
        )
        indices = indices[~out_of_bounds]

        # TODO look up neatest way of vectorizing xarrays - this works tho
        ind_along_z = xr.DataArray(indices[:, 0], dims=["new_index"])
        ind_along_y = xr.DataArray(indices[:, 1], dims=["new_index"])
        ind_along_x = xr.DataArray(indices[:, 2], dims=["new_index"])

        space[ind_along_z, ind_along_y, ind_along_x] += self.intensities[
            ~out_of_bounds
        ]
        return space
