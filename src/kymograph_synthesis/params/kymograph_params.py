from typing import Optional, Literal
from pydantic import BaseModel, ConfigDict


class KymographParams(BaseModel):

    model_config = ConfigDict(validate_assignment=True)

    sample_path_points: list[tuple[float, float, float]]

    path_clip: tuple[float, float]  # TODO: validate between 0 and 1

    interpolation: Optional[
        Literal[
            "linear",
            "nearest",
            "nearest-up",
            "zero",
            "slinear",
            "quadratic",
            "cubic",
            "previous",
            "next",
        ]
    ]
