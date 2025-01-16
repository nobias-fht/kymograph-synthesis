from pydantic import BaseModel, ConfigDict, Field
from .fluorophone_distributions import FluoroDistrParams, SimplexNoiseParams

import numpy as np


class RenderingParams(BaseModel):

    model_config = ConfigDict(validate_assignment=True, validate_default=True)

    particle_path_points: list[tuple[float, float, float]]

    """Points that create the path the particles are rendered on."""
    static_distributions: list[FluoroDistrParams] = Field(
        default=[
            SimplexNoiseParams(
                scales=[5, 10],
                max_fluorophore_count_per_nm3=np.random.uniform(0.02, 0.04),
            )
        ]
    )
