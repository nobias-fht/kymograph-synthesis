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
        # Squeezing because thickness parameter adds new dimension
        space_coords = static_path(path_positions).squeeze() 
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

        # have to use unbuffered add,
        # for particles in the same place the values will be accumulated
        space_numpy = np.zeros(space.shape)
        np.add.at(
            space_numpy,
            tuple(indices[:, i] for i in range(3)),
            self.intensities[~out_of_bounds],
        )

        space += xp.asarray(space_numpy)
        return space
