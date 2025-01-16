from typing import Generator

import numpy as np
from numpy.typing import NDArray
import microsim.schema as ms

from ..render.fluorophore_distributions import ParticleSystem
from ..params import ImagingParams


def generate_image_time_series(
    params: ImagingParams,
    particle_system_generator: Generator[ParticleSystem, None, None],
    static_distributions: list[ms.FluorophoreDistribution],
) -> NDArray:
    # also not having access to number of steps here means have to use concat
    frames = [] 
    for particle_system in particle_system_generator:
        sim = ms.Simulation(
            **params.model_dump(),
            sample=static_distributions + [particle_system]
        )
        frames.append(sim.digital_image())
    return np.concatenate(frames)
