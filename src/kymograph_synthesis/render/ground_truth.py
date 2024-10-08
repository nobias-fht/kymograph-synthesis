from typing import Any

import numpy as np
from numpy.typing import NDArray

from .static_path import StaticPath, LinearPath
from ..dynamics.system import gen_simulation_data


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

def render_linear(
    resolution: tuple[int, ...],
    path_start: tuple[int, ...],
    path_end: tuple[int, ...],
    n_particles: int,
    n_frames: int,
    **dynamics_simulation_kwargs: Any
):

    particle_positions = gen_simulation_data(
        n_particles, n_frames, **dynamics_simulation_kwargs
    )

    space = np.zeros((n_frames, *resolution), dtype=float)
    linear_path = LinearPath(start=np.array(path_start), end=np.array(path_end))
    space_coords = linear_path(particle_positions)

    space_indicies = np.round(space_coords).astype(int)

    for i in range(n_frames):
        frame_coords = space_indicies[i, :, :]
        out_of_bounds = (frame_coords < 0) | (frame_coords >= np.array(resolution).reshape(1, -1))
        frame_coords = frame_coords[~out_of_bounds.any(axis=1), :]
        for particle_index in range(frame_coords.shape[0]):
            m, n = frame_coords[particle_index, :]
            space[i, m, n] += 1

    return space
