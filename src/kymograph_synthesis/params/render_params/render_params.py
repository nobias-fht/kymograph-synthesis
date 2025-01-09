from pydantic import BaseModel, ConfigDict, Field
from .fluorophone_distributions import FluorphoneDistributionParams, SimplexNoiseParams

import numpy as np


class RenderingParams(BaseModel):

    model_config = ConfigDict(validate_assignment=True, validate_default=True)

    particle_path_points: list[tuple[float, float, float]]
    """Points that create the path the particles are rendered on."""

    static_distributions: list[FluorphoneDistributionParams] = Field(
        default=[
            SimplexNoiseParams(
                noise_scales=[0.5, 0.25],
                scale_weights=[1, 0.5],
                max_intensity=np.random.uniform(300, 600) * 10e-4,
            )
        ]
    )
