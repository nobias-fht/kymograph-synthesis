from typing import Generator

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import interp1d

from kymograph_synthesis.render.static_path import PiecewiseQuadraticBezierPath


def generate_state_ground_truth(
    particle_positions: NDArray[np.float64],
    particle_states: NDArray[np.int_],
    n_spatial_values: int,
    path_points: list[tuple[float, float, float]],
    *,
    line_thickness: int,
    project_to_xy: bool,
):
    if project_to_xy:
        particle_positions = _project_positions_to_xy(
            particle_positions, path_points, n_spatial_values
        )

    n_steps = particle_positions.shape[0]

    kymograph_gt = np.zeros((n_steps, n_spatial_values, 3), dtype=np.uint8)
    particle_positions_pixels = particle_positions * n_spatial_values
    for p in range(particle_positions.shape[1]):
        indices_gen = _calc_line_indices(
            particle_positions_pixels[:, p],
            n_spatial_values,
            thickness=line_thickness,
        )
        for t, indices in enumerate(indices_gen):
            kymograph_gt[t, indices, particle_states[t, p]] = 1

    return kymograph_gt


def generate_instance_ground_truth(
    particle_positions: NDArray[np.float64],
    n_spatial_values: int,
    path_points: list[tuple[float, float, float]],
    *,
    line_thickness: int,
    project_to_xy: bool,
):
    if project_to_xy:
        particle_positions = _project_positions_to_xy(
            particle_positions, path_points, n_spatial_values
        )
    n_steps, n_particles = particle_positions.shape
    kymograph_gt = np.zeros(((n_steps, n_spatial_values, n_particles)), dtype=np.uint8)
    particle_positions_pixels = particle_positions * n_spatial_values
    for p in range(particle_positions.shape[1]):
        indices_gen = _calc_line_indices(
            particle_positions_pixels[:, p],
            n_spatial_values,
            thickness=line_thickness,
        )
        for t, indices in enumerate(indices_gen):
            kymograph_gt[t, indices, p] = 1
    return kymograph_gt


def _project_positions_to_xy(
    particle_positions: NDArray[np.float64],
    path_points: list[tuple[float, float, float]],
    n_spatial_values: int,
) -> NDArray[np.float64]:
    static_path = PiecewiseQuadraticBezierPath(
        [np.array(point) for point in path_points]
    )

    sample_points = static_path(np.linspace(0, 1, n_spatial_values))
    d = sample_points[1:] - sample_points[:-1]
    mag = np.linalg.norm(d, axis=-1)
    adjacent = (d[..., 2] ** 2 + d[..., 1] ** 2) ** 0.5
    theta = np.arctan2(d[..., 0], adjacent)
    projected_mag = mag * np.cos(theta)

    interp_f = interp1d(
        np.linspace(0, 1, n_spatial_values),
        np.concatenate([[0], np.cumsum(projected_mag) / np.sum(projected_mag)]),
        fill_value="extrapolate",
    )
    new_positions = interp_f(particle_positions)
    return new_positions


def _calc_line_indices(
    particle_pixel_positions: NDArray[np.floating],
    n_spatial_values: int,
    thickness: float = 1,
) -> Generator[NDArray[np.int_], None, None]:
    mid_points = (particle_pixel_positions[1:] + particle_pixel_positions[:-1]) / 2
    half_width = thickness - 1
    n_steps = len(particle_pixel_positions)
    prev_indices: list[NDArray[np.int_]] | None = None
    for t in range(n_steps - 1):
        x0 = particle_pixel_positions[t]
        x1 = mid_points[t]
        x2 = particle_pixel_positions[t + 1]

        d = np.sign(x2 - x1)
        if d == 0:
            d = 1

        current_indices: list[NDArray[np.int_]] = []
        for points in [(x0, x1), (x1, x2)]:
            p0 = np.round(points[0] - half_width * d)
            p1 = np.round(points[1] + half_width * d)
            if p1 == p0:
                p1 += d

            indices = np.arange(p0, p1, d).astype(int)
            in_bounds = np.logical_and((0 <= indices), (indices < n_spatial_values))
            current_indices.append(indices[in_bounds])

        if prev_indices is None:
            yield current_indices[0]
        else:
            yield np.concatenate([prev_indices[-1], current_indices[0]])

        prev_indices = current_indices

    if prev_indices is not None:
        yield prev_indices[-1]
