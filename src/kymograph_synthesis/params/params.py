from typing import Any

import numpy as np
from pydantic import (
    ConfigDict,
    BaseModel,
    ValidationInfo,
    model_validator,
    field_validator,
)

from .render_params import RenderingParams
from .dynamics_params import DynamicsParams
from .kymograph_params import KymographParams



class Params(BaseModel):

    model_config = ConfigDict(validate_assignment=True, validate_default=True)

    dynamics: DynamicsParams = DynamicsParams()

    rendering: RenderingParams = RenderingParams()

    kymograph: KymographParams


    @field_validator("kymograph", mode="before")
    @classmethod
    def set_kymograph_defaults(cls, value: Any, info: ValidationInfo) -> Any:
        if not isinstance(value, dict):
            return value

        rendering_params = info.data.get("rendering")
        assert isinstance(rendering_params, RenderingParams)
        imaging_params = rendering_params.imaging

        truth_space_shape = imaging_params.truth_space.shape
        truth_space_scale = imaging_params.truth_space.scale
        downscale = imaging_params.output_space.downscale
        particle_path_points = rendering_params.particle_path_points

        kymo_sample_path_points = calc_kymo_sample_path(
            particle_path_points,
            truth_space_scale,
            downscale,
            truth_space_shape[0] // 2,
        )
        value.setdefault("sample_path_points", kymo_sample_path_points)
        return value
    
    @model_validator(mode="before")
    @classmethod
    def set_empty_values(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return
        
        data.setdefault("kymograph", {})
        return data

# TODO: move to helper module
def calc_kymo_sample_path(
    particle_path_points: list[tuple[float, float, float]],
    truth_space_scale: tuple[float, float, float],
    output_space_downscale: int,
    truth_space_z_idx: int,
) -> list[tuple[float, float, float]]:
    kymo_sample_path_points = np.array(
        [
            (np.array(point) / np.array(truth_space_scale)) / output_space_downscale
            for point in particle_path_points
        ]
    )
    kymo_sample_path_points[:, 0] = truth_space_z_idx / output_space_downscale
    return [tuple(point) for point in kymo_sample_path_points]


