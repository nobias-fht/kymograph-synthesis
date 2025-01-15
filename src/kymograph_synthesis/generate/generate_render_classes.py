from typing import Generator

import numpy as np
from numpy.typing import NDArray

from ..params.render_params import RenderingParams, FluoroDistrName
from ..render.fluorophore_distributions import SimplexNoise, ParticleSystem
from ..render.static_path import PiecewiseQuadraticBezierPath

def generate_render_classes(
    params: RenderingParams,
    particle_positions: NDArray[np.float_],
    particle_fluorophore_count: NDArray[np.float_],
    # TODO: this is annoying, doesn't fit with the asthetic :(
    truth_space_shape: tuple[int, int, int],
    truth_space_scale: tuple[float, float, float],
):
    static_distributions = [
        fluoro_distr_factory(**fluoro_distr_params)
        for fluoro_distr_params in params.static_distributions
    ]


def particle_system_generator(
    particle_path_points: list[tuple[float, float, float]],
    particle_positions: NDArray[np.float_],
    particle_fluorophore_count: NDArray[np.float_],
    # TODO: this is annoying, doesn't fit with the asthetic :(
    #   - path points should be in um or deal with it?
    truth_space_shape: tuple[int, int, int],
    truth_space_scale: tuple[float, float, float],
) -> Generator[ParticleSystem, None, None]:
    particle_path_points_um = [
        np.array(point)
        * np.array(truth_space_shape)
        * np.array(truth_space_scale)
        for point in particle_path_points
    ]
    static_path = PiecewiseQuadraticBezierPath(particle_path_points_um)
    n_steps = particle_positions.shape[0]
    for t in range(n_steps):
        yield ParticleSystem.on_static_path(
            static_path,
            particle_positions[t],
            particle_fluorophore_count[t],
        )


def fluoro_distr_factory(*, name: FluoroDistrName, **kwargs):
    match name:
        case FluoroDistrName.SIMPLEX_NOISE:
            return SimplexNoise(**kwargs)
        case _:
            raise ValueError(f"Unrecognised fluorophore distribution name '{name}'.")
