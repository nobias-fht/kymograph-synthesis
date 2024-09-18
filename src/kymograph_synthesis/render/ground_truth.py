from typing import Any

import numpy as np

from .static_path import StaticPath, LinearPath
from ..dynamics.system import gen_simulation_data


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
        frame_coords = space_indicies[:, i, :]
        out_of_bounds = (frame_coords < 0) | (frame_coords >= np.array(resolution).reshape(-1, 1))
        frame_coords = frame_coords[:, ~out_of_bounds.any(axis=0)]
        for particle_index in range(frame_coords.shape[1]):
            m, n = frame_coords[:, particle_index]
            space[i, m, n] += 1

    return space


