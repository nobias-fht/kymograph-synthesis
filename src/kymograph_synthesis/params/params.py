from typing import Any

import numpy as np
from numpy.typing import NDArray
from pydantic import ConfigDict, BaseModel, Field, model_validator, TypeAdapter
import microsim.schema as ms

from .render_params import RenderingParams
from .dynamics_params import DynamicsParams
from .kymograph_params import KymographParams


# microsim params but exclude sample, ParticleSystem and RenderingParams are elsewhere
class ImagingParams(ms.Simulation):

    # use default factory to avoid same reference to labels list in each instance
    # (even though this is here as a dummy var)
    sample: ms.Sample = Field(
        default_factory=lambda: ms.Sample(labels=[]), exclude=True
    )

    _truth_space_scale_default: tuple[float, float, float] = (0.04, 0.01, 0.01)

    truth_space: ms.ShapeScaleSpace = ms.ShapeScaleSpace(
        shape=(32, 64, 512), scale=_truth_space_scale_default
    )

    output_space: ms.space.Space = Field(default=ms.DownscaledSpace(downscale=4))

    modality: ms.Modality = Field(default=ms.Widefield())

    detector: ms.detectors.Detector = Field(
        default=ms.CameraCCD(qe=1, read_noise=4, bit_depth=12, offset=100)
    )

    exposure_ms: float = 200

    @model_validator(mode="before")
    @classmethod
    def set_truth_space_scale_default(cls, data: dict[str, Any]) -> dict[str, Any]:
        non_default_truth_space = "truth_space" in data
        if non_default_truth_space and isinstance(data["truth_space"], dict):
            if "scale" not in data["truth_space"]:
                data["truth_space"]["scale"] = cls._truth_space_scale_default.default
        return data


# TODO: move truth space shape to rendering?


def _kymograph_sample_path_points_default_factory(
    data: dict[str, Any]
) -> list[tuple[float, float, float]]:
    assert isinstance(data["rendering"], RenderingParams)
    assert isinstance(data["imaging"], ImagingParams)

    truth_space_shape = np.array(data["imaging"].truth_space.shape)
    z_dims = truth_space_shape[0]
    z_mid = z_dims / 2
    # relative path points are expressed as a ratio of the 
    relative_particle_path_points: NDArray[np.float_] = np.array(
        data["rendering"].particle_path_points, dtype=float
    )
    # path_points in pixel index space coordinates
    particle_path_points: NDArray[np.float_] = (
        relative_particle_path_points * truth_space_shape
    )
    sample_path_points = particle_path_points.copy()
    # replace z with the mid point of the stack for sampling
    sample_path_points[:, 0] = z_mid
    sample_path_points = sample_path_points / data["imaging"].output_space.downscale
    return [(zi, yi, xi) for zi, yi, xi in sample_path_points]


def _kymograph_params_default_factory(data: dict[str, Any]) -> KymographParams:
    sample_path_points = _kymograph_sample_path_points_default_factory(data)
    return KymographParams(sample_path_points=sample_path_points)


class Params(BaseModel):

    model_config = ConfigDict(validate_assignment=True, validate_default=True)

    dynamics: DynamicsParams = DynamicsParams()

    rendering: RenderingParams = RenderingParams()

    imaging: ImagingParams = ImagingParams()

    kymograph: KymographParams = Field(
        default_factory=_kymograph_params_default_factory
    )
