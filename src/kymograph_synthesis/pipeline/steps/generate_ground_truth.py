from typing import TypedDict

import numpy as np
from numpy.typing import NDArray


class GenerateGroundTruthOutput(TypedDict):
    ground_truth: NDArray


def generate_ground_truth(
    particle_positions: NDArray[np.float_],
    particle_states: NDArray[np.int_],
    n_spatial_values: int,
):
    n_steps = particle_positions.shape[0]

    kymograph_gt = np.zeros((n_steps, n_spatial_values, 3))
    particle_positions_pixels = particle_positions * n_spatial_values
    mid_points = (particle_positions_pixels[1:] + particle_positions_pixels[:-1]) / 2
    for p in range(particle_positions.shape[1]):
        for t in range(n_steps):
            if t == 0:
                x0 = particle_positions_pixels[t, p]
            else:
                x0 = mid_points[t - 1, p]

            x1 = particle_positions_pixels[t, p]

            if t == n_steps - 1:
                x2 = particle_positions_pixels[t, p]
            else:
                x2 = mid_points[t, p]

            d1 = np.sign(np.floor(x1) - np.floor(x0))
            d2 = np.sign(np.floor(x2) - np.floor(x1))
            indices = np.concatenate(
                [
                    np.array([int(np.floor(x0))]),
                    np.array([int(np.floor(x1))]),
                    np.arange(np.floor(x0), np.floor(x1), d1 if d1 != 0 else 1).astype(
                        int
                    ),
                    np.arange(np.floor(x1), np.floor(x2), d2 if d2 != 0 else 1).astype(
                        int
                    ),
                ]
            )
            in_bounds = np.logical_and((0 <= indices), (indices < n_spatial_values))
            kymograph_gt[t, indices[in_bounds], particle_states[t, p]] = 1

    return GenerateGroundTruthOutput(ground_truth=kymograph_gt)
