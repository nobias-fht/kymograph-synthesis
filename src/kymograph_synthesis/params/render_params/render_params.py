from collections.abc import Sequence
from typing import Any
from functools import partial

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
    ValidationInfo,
)
import numpy as np

from .fluorophone_distributions import FluoroDistrParams, SimplexNoiseParams
from .imaging_params import ImagingParams


def _particle_path_points_default_factory(
    data: dict[str, Any], n_points: int
) -> Sequence[tuple[float, float, float]]:
    imaging_params = data["imaging"]
    assert isinstance(imaging_params, ImagingParams)
    truth_space_shape = imaging_params.truth_space.shape
    truth_space_scale = imaging_params.truth_space.scale
    relative_path_points = _random_relative_particle_path_points(n_points)
    return _convert_relative_to_um(
        relative_path_points, truth_space_shape, truth_space_scale
    )


class RenderingParams(BaseModel):

    model_config = ConfigDict(validate_assignment=True, validate_default=True)

    _default_truth_space_shape: tuple[int, int, int] = (16, 32, 512)

    _default_truth_space_scale: tuple[float, float, float] = (0.04, 0.01, 0.01)

    imaging: ImagingParams

    particle_path_points: Sequence[tuple[float, float, float]] = Field(
        default_factory=lambda data: _particle_path_points_default_factory(
            data, n_points=4
        )
    )
    """Points that create the path the particles are rendered on."""
    
    static_distributions: list[FluoroDistrParams] = Field(
        default=[
            SimplexNoiseParams(
                scales=[5, 10],
                max_fluorophore_count_per_nm3=np.random.uniform(0.02, 0.04),
            )
        ]
    )

    @field_validator("imaging", mode="before")
    @classmethod
    def set_imaging_defaults(cls, value: Any, info: ValidationInfo) -> Any:
        if not isinstance(value, dict):
            return value

        value.setdefault("truth_space", {})
        truth_space = value.get("truth_space")
        if not isinstance(truth_space, dict):
            return value

        default_truth_space_shape = cls._default_truth_space_shape.default
        default_truth_space_scale = cls._default_truth_space_scale.default
        truth_space.setdefault("shape", default_truth_space_shape)
        truth_space.setdefault("scale", default_truth_space_scale)
        return value

    @model_validator(mode="before")
    @classmethod
    def set_empty_values(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return

        data.setdefault("imaging", {})
        return data


def _random_relative_particle_path_points(
    n_points: int,
) -> list[tuple[float, float, float]]:
    x = np.zeros(n_points, dtype=float)
    x[0] = np.random.uniform(0.1, 0.3)
    x[1:] = np.sort(np.random.uniform(x[0], 0.9, n_points - 1))
    if x[-1] - x[0] < 0.5:
        x[-1] = np.random.uniform(x[0] + 0.5, 0.9)

    y = np.random.uniform(0.1, 0.9, n_points)
    z = np.random.normal(0.5, 0.15, n_points)
    z_out_of_frame = np.logical_or(z < 0, z > 1)
    z[z_out_of_frame] = np.random.uniform(0.05, 0.95, np.count_nonzero(z_out_of_frame))

    anisotropic_points = np.array([(zi, yi, xi) for zi, yi, xi in zip(z, y, x)])

    return [(zi, yi, xi) for zi, yi, xi in anisotropic_points]


def _convert_relative_to_um(
    relative_points: list[tuple[float, float, float]],
    truth_space_shape: tuple[int, int, int],
    truth_space_scale: tuple[float, float, float],
) -> Sequence[tuple[float, float, float]]:
    um_points = [
        np.array(point) * np.array(truth_space_shape) * np.array(truth_space_scale)
        for point in relative_points
    ]
    return [tuple(point) for point in um_points]
