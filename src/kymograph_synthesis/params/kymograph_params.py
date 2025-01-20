from typing import Optional, Literal
from pydantic import BaseModel, ConfigDict, field_validator


class KymographParams(BaseModel):

    model_config = ConfigDict(validate_assignment=True)

    sample_path_points: list[tuple[float, float, float]]

    n_spatial_values_factor: float = 1.2

    interpolation: Literal[
        "none",
        "linear",
        "nearest",
        "nearest-up",
        "zero",
        "slinear",
        "quadratic",
        "cubic",
        "previous",
        "next",
    ] = "cubic"
