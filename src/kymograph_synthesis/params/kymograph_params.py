from typing import Optional, Literal
from pydantic import BaseModel, ConfigDict, field_validator


class KymographParams(BaseModel):

    model_config = ConfigDict(validate_assignment=True)

    sample_path_points: list[tuple[float, float, float]]

    path_clip: tuple[float, float]=[0.1, 0.9]

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
    ] = "cubic"

    @field_validator("path_clip")
    @classmethod
    def validate_path_clip(cls, value: tuple[float, float]):
        for i, v in enumerate(value):
            if (0 > v) or (v > 1):
                raise ValueError(
                    f"Values for `path_clip` must be in [0, 1], found value {v} at "
                    f"position {i}."
                )
        return value