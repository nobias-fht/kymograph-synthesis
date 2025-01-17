from typing import cast
import numpy as np
import xarray as xr
from numpy.typing import NDArray
import microsim.schema as ms
from microsim._data_array import xrDataArray

from ..static_path import StaticPath


class ParticleSystem:

    def __init__(self, coords: NDArray, fluorophore_counts: NDArray):

        # Coords are in real units - um, only positive number quadrant is rendered
        # TODO: make sure dimensions match
        self.coords = coords  # N x D
        self.intensities = fluorophore_counts  # N

        self.n_particles = self.coords.shape[1]

    @classmethod
    def on_static_path(
        cls,
        static_path: StaticPath,
        path_positions: NDArray,
        fluorophore_counts: NDArray,
    ):
        # static path points have to be defined in real world coordinates
        space_coords = static_path(path_positions)
        return cls(coords=space_coords, fluorophore_counts=fluorophore_counts)

    def render(self, space: xrDataArray, xp: ms.NumpyAPI | None = None) -> xrDataArray:
        truth_space = cast(ms.space.Space, space.attrs["space"])
        space_coords = self.coords / np.array(truth_space.scale)

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

        space[ind_along_z, ind_along_y, ind_along_x] += self.intensities[~out_of_bounds]
        return space
