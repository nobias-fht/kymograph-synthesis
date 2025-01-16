from typing import Any

import numpy as np
from pydantic import (
    ConfigDict,
    BaseModel,
    Field,
    ValidationInfo,
    model_validator,
    field_validator,
)
import microsim.schema as ms

from .render_params import RenderingParams
from .dynamics_params import DynamicsParams
from .kymograph_params import KymographParams


# TODO: move imaging params to own module
# microsim params but exclude sample, ParticleSystem and RenderingParams are elsewhere
class ImagingParams(ms.Simulation):

    # use default factory to avoid same reference to labels list in each instance
    # (even though this is here as a dummy var)
    sample: ms.Sample = Field(
        default_factory=lambda: ms.Sample(labels=[]), exclude=True
    )

    output_space: ms.space.Space = Field(default=ms.DownscaledSpace(downscale=4))

    modality: ms.Modality = Field(default=ms.Widefield())

    detector: ms.detectors.Detector = Field(
        default=ms.CameraCCD(qe=1, read_noise=4, bit_depth=12, offset=100)
    )

    exposure_ms: float = 200


class Params(BaseModel):

    model_config = ConfigDict(validate_assignment=True, validate_default=True)

    _default_truth_space_shape: tuple[int, int, int] = (16, 32, 512)

    _default_truth_space_scale: tuple[float, float, float] = (0.04, 0.01, 0.01)

    dynamics: DynamicsParams = DynamicsParams()

    imaging: ImagingParams

    rendering: RenderingParams

    kymograph: KymographParams

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

    @field_validator("rendering", mode="before")
    @classmethod
    def set_rendering_defaults(cls, value: Any, info: ValidationInfo) -> Any:
        if not isinstance(value, dict):
            return value
        
        imaging_params = info.data.get("imaging")
        assert isinstance(imaging_params, ImagingParams)
        truth_space_shape = imaging_params.truth_space.shape
        truth_space_scale = imaging_params.truth_space.scale

        relative_path_points = _random_relative_particle_path_points()
        path_points = _convert_relative_to_um(
            relative_path_points, truth_space_shape, truth_space_scale
        )
        value.setdefault("particle_path_points", path_points)
        return value


    @field_validator("kymograph", mode="before")
    @classmethod
    def set_kymograph_defaults(cls, value: Any, info: ValidationInfo) -> Any:
        if not isinstance(value, dict):
            return value

        imaging_params = info.data.get("imaging")
        assert isinstance(imaging_params, ImagingParams)
        rendering_params = info.data.get("rendering")
        assert isinstance(rendering_params, RenderingParams)

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
        
        data.setdefault("imaging", {})
        data.setdefault("rendering", {})
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


def _random_relative_particle_path_points() -> list[tuple[float, float, float]]:
    n_points = 5
    x = np.zeros(n_points, dtype=float)
    x[0] = np.random.uniform(0.1, 0.4)
    x[1:] = np.sort(np.random.uniform(x[0], 0.9, n_points - 1))
    if x[3] - x[0] < 0.4:
        x[3] = np.random.uniform(x[0] + 0.4, 0.9)

    y = np.random.uniform(0.1, 0.9, n_points)
    z = np.random.normal(0.5, 0.1, n_points)

    anisotropic_points = np.array([(zi, yi, xi) for zi, yi, xi in zip(z, y, x)])

    return [(zi, yi, xi) for zi, yi, xi in anisotropic_points]


def _convert_relative_to_um(
    relative_points: list[tuple[float, float, float]],
    truth_space_shape: tuple[int, int, int],
    truth_space_scale: tuple[float, float, float],
) -> list[float, float, float]:
    um_points = [
        np.array(point) * np.array(truth_space_shape) * np.array(truth_space_scale)
        for point in relative_points
    ]
    return [tuple(point) for point in um_points]
