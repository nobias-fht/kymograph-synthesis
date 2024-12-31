from pydantic import BaseModel, ConfigDict
from .fluorophone_distributions import FluorphoneDistributions 

class RenderParams(BaseModel):

    model_config = ConfigDict(validate_assignment=True)

    particle_path_points: list[tuple[float, float, float]]
    """Points that create the path the particles are rendered on."""

    static_distributions: list[FluorphoneDistributions]