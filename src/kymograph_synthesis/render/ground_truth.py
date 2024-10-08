from typing import Any

import numpy as np
from numpy.typing import NDArray

from .static_path import StaticPath, LinearPath


def render(
    resolution: tuple[int, ...],
    static_path: StaticPath,
    positions: NDArray,
    intensities: NDArray,
):
    # TODO: check positions and intensities have the same shape
    n_frames = positions.shape[0]
    space_time = np.zeros((n_frames, *resolution))
    space_coords = static_path(positions)
    space_indicies = np.round(space_coords).astype(int)

    for t in range(n_frames):
        frame_coords = space_indicies[t, :, :]
        for particle_index in range(frame_coords.shape[0]):
            i, j = frame_coords[particle_index, :]
            if i < 0 or j < 0:
                continue
            if (np.array([i, j]) >= np.array(resolution)).any():
                continue
            space_time[t, i, j] += intensities[t, particle_index]
    return space_time

