from typing import Generator

import numpy as np
from numpy.typing import NDArray

from kymograph_synthesis.render.static_path import PiecewiseQuadraticBezierPath


def generate_state_ground_truth(
    particle_positions: NDArray[np.float_],
    particle_states: NDArray[np.int_],
    n_spatial_values: int,
    path_points: list[tuple[float, float, float]],
    *,
    line_thickness: int,
    project_to_xy: bool,
):
    if project_to_xy:
        particle_positions = _project_positions_to_xy(particle_positions, path_points)

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
    particle_positions: NDArray[np.float_],
    n_spatial_values: int,
    path_points: list[tuple[float, float, float]],
    *,
    line_thickness: int,
    project_to_xy: bool,
):
    if project_to_xy:
        particle_positions = _project_positions_to_xy(particle_positions, path_points)
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
    particle_positions: NDArray[np.float_],
    path_points: list[tuple[float, float, float]],
) -> NDArray[np.float_]:
    static_path = PiecewiseQuadraticBezierPath(
        [np.array(point) for point in path_points]
    )
    projected_path_points = []
    for point in path_points:
        point = (0, *point[1:])
        projected_path_points.append(np.array(point))
    projected_path = static_path

    particle_coords = static_path(particle_positions).squeeze()
    velocity = particle_coords[1:] - particle_coords[:-1]

    adjacent = (velocity[..., 2] ** 2 + velocity[..., 1] ** 2) ** 0.5
    theta = np.arctan2(velocity[..., 0], adjacent)
    speed = np.linalg.norm(velocity, axis=-1)
    projected_speed = speed * np.cos(theta)
    direction = np.sign(particle_positions[1:] - particle_positions[:-1])

    # have to rescale positions because of inaccurate calculations for out of bounds movement
    initial_positions = particle_positions[0]
    new_position = np.tile(initial_positions, (particle_positions.shape[0], 1))
    new_position[1:] = new_position[1:] + np.cumsum(
        projected_speed * direction / projected_path.length(), axis=0
    )
    in_bounds = np.logical_and(0 <= particle_positions, particle_positions <= 1)
    for p in range(particle_positions.shape[1]):
        if not in_bounds[:, p].any():
            continue
        target_min = particle_positions[:, p][in_bounds[:, p]].min()
        target_max = particle_positions[:, p][in_bounds[:, p]].max()
        min_ = new_position[:, p][in_bounds[:, p]].min()
        max_ = new_position[:, p][in_bounds[:, p]].max()

        new_position[:, p] = (
            (new_position[:, p] * (target_max - target_min) / (max_ - min_))
            - min_
            + target_min
        )
    return new_position


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
