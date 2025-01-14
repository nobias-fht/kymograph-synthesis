from typing import Any

from pydantic import BaseModel, ConfigDict, Field
from .fluorophone_distributions import FluorphoneDistributionParams, SimplexNoiseParams

import numpy as np


def _render_particle_path_points_factory(
) -> list[tuple[float, float, float]]:
    n_points = 4
    x = np.zeros(n_points, dtype=float)
    x[0] = 0.1
    x[1:-1] = np.sort(np.random.uniform(0.2, 0.8, n_points - 2))
    x[-1] = 0.9
    y = np.random.uniform(0.1, 0.9, n_points)
    z = np.random.uniform(0.1, 0.9, n_points)

    anisotropic_points = np.array([(zi, yi, xi) for zi, yi, xi in zip(z, y, x)])

    return [(zi, yi, xi) for zi, yi, xi in anisotropic_points]


class RenderingParams(BaseModel):

    model_config = ConfigDict(validate_assignment=True, validate_default=True)

    particle_path_points: list[tuple[float, float, float]] = Field(
        default_factory=_render_particle_path_points_factory
    )

    """Points that create the path the particles are rendered on."""
    static_distributions: list[FluorphoneDistributionParams] = Field(
        default=[
            SimplexNoiseParams(
                scales=[5, 10], max_fluorophore_count_per_nm3=np.random.uniform(0.02, 0.04)
            )
        ]
    )
