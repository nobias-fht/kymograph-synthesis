import numpy as np
from numpy.typing import NDArray

# TODO: fix this mess
def generate_state_ground_truth(
    particle_positions: NDArray[np.float_],
    particle_states: NDArray[np.int_],
    n_spatial_values: int,
    line_thickness: int,
):
    if (line_thickness - 1) % 2 != 0:
        raise ValueError("Only odd values for line thickness are supported.")
    padding = (line_thickness - 1) // 2
    n_steps = particle_positions.shape[0]

    kymograph_gt = np.zeros((n_steps, n_spatial_values, 3), dtype=np.uint8)
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
                    np.arange(
                        int(np.floor(x0)) - padding,
                        int(np.floor(x0)) + padding + 1,
                        dtype=int,
                    ),
                    np.arange(
                        int(np.floor(x1)) - padding,
                        int(np.floor(x1)) + padding + 1,
                        dtype=int,
                    ),
                    np.arange(
                        np.floor(x0) - d1 * padding,
                        np.floor(x1) + d1 * padding,
                        d1 if d1 != 0 else 1,
                        dtype=int,
                    ),
                    np.arange(
                        np.floor(x1) - d2 * padding,
                        np.floor(x2) + d2 * padding,
                        d2 if d2 != 0 else 1,
                        dtype=int,
                    ),
                ]
            )
            in_bounds = np.logical_and((0 <= indices), (indices < n_spatial_values))
            kymograph_gt[t, indices[in_bounds], particle_states[t, p]] = 1

    return kymograph_gt


def generate_instance_ground_truth(
    particle_positions: NDArray[np.float_],
    n_spatial_values: int,
    line_thickness: int,
):
    if (line_thickness - 1) % 2 != 0:
        raise ValueError("Only odd values for line thickness are supported.")
    padding = (line_thickness - 1) // 2

    n_steps, n_particles = particle_positions.shape
    kymograph_gt = np.zeros(((n_steps, n_spatial_values, n_particles)), dtype=np.uint8)
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
                np.arange(int(np.floor(x0))-padding, int(np.floor(x0))+padding+1, dtype=int),
                np.arange(int(np.floor(x1))-padding, int(np.floor(x1))+padding+1, dtype=int),
                np.arange(np.floor(x0)-d1*padding, np.floor(x1)+d1*padding, d1 if d1 != 0 else 1, dtype=int),
                np.arange(np.floor(x1)-d2*padding, np.floor(x2)+d2*padding, d2 if d2 != 0 else 1, dtype=int)
            ]
        )
            in_bounds = np.logical_and((0 <= indices), (indices < n_spatial_values))
            kymograph_gt[t, indices[in_bounds], p] = 1
    return kymograph_gt
