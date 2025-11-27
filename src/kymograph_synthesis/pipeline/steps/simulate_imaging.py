from typing import TypedDict

import numpy as np
from numpy.typing import NDArray
import microsim.schema as ms
from microsim.schema.sample import AnyDistribution

from ...params.render_params import (
    RenderingParams,
    FluoroDistrName,
    FluoroDistrParams,
)
from ...render.fluorophore_distributions import SimplexNoise, ParticleSystem
from ...render.static_path import PiecewiseQuadraticBezierPath


class ImagingSimOutput(TypedDict):
    frames: NDArray[np.uint16]
    path_length_um: float


def simulate_imaging(
    params: RenderingParams,
    n_steps: int,
    particle_positions: NDArray[np.float_],
    particle_fluorophore_count: NDArray[np.float_],
) -> ImagingSimOutput:
    seed = params.imaging.settings.random_seed
    if seed is None:
        raise ValueError("Random seed for microsim must be set.")
    rng = np.random.default_rng(seed)

    static_distributions = initialize_static_distributions(params.static_distributions)
    output_space_shape = params.imaging.output_space.shape

    particle_path_points_um = [np.array(point) for point in params.particle_path_points]
    static_path = PiecewiseQuadraticBezierPath(particle_path_points_um)

    frames = np.zeros((n_steps, *output_space_shape), dtype=np.uint16)
    for t in range(n_steps):
        particle_system = initialize_particle_system(
            time_point=t,
            static_path=static_path,
            particle_positions=particle_positions,
            particle_fluorophore_count=particle_fluorophore_count,
        )
        # copy so that saved seed won't be the seed + t
        settings = params.imaging.settings.model_copy()
        settings.random_seed = rng.integers(0, 2**5)
        fluro_distributions = [
            ms.FluorophoreDistribution(distribution=distr)
            for distr in static_distributions
        ]
        fluro_distributions.append(
            ms.FluorophoreDistribution(distribution=particle_system)
        )
        sim = ms.Simulation(
            **params.imaging.model_dump(exclude={"settings"}),
            settings=settings,
            sample=ms.Sample(labels=fluro_distributions),
        )
        frames[t] = sim.digital_image().data.get()
    return ImagingSimOutput(frames=frames, path_length_um=static_path.length())


def fluoro_distr_factory(*, name: FluoroDistrName, **kwargs) -> AnyDistribution:
    match name:
        case FluoroDistrName.SIMPLEX_NOISE:
            return SimplexNoise(**kwargs)
        case _:
            raise ValueError(f"Unrecognised fluorophore distribution name '{name}'.")


def initialize_static_distributions(
    static_distributions: list[FluoroDistrParams],
) -> list[AnyDistribution]:
    return [
        fluoro_distr_factory(**fluoro_distr_params.model_dump())
        for fluoro_distr_params in static_distributions
    ]


def initialize_particle_system(
    time_point: int,
    static_path: PiecewiseQuadraticBezierPath,
    particle_positions: NDArray[np.float_],
    particle_fluorophore_count: NDArray[np.float_],
) -> ParticleSystem:
    return ParticleSystem.on_static_path(
        static_path,
        particle_positions[time_point],
        particle_fluorophore_count[time_point],
    )
