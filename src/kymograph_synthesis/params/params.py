from typing import Any

import numpy as np
from numpy.typing import NDArray
from pydantic import ConfigDict, BaseModel, Field
import microsim.schema as ms

from .render_params import RenderingParams
from .dynamics_params import DynamicsParams
from .kymograph_params import KymographParams

# microsim params but exclude sample, ParticleSystem and RenderingParams are elsewhere
class ImagingParams(ms.Simulation):

    sample: ms.Sample = Field(default=ms.Sample(labels=[]), exclude=True)


def _render_particle_path_points_factory(
    data: dict[str, Any]
) -> list[tuple[float, float, float]]:
    truth_space_shape = np.array(data["imaging"].truth_space.shape)

    n_points = 4
    x = np.zeros(n_points, dtype=float)
    x[0] = 0.1
    x[1:-1] = np.sort(np.random.uniform(0.2, 0.8, n_points - 2))
    x[-1] = 0.9
    y = np.random.uniform(0.1, 0.9, n_points)
    z = np.random.uniform(0.1, 0.9, n_points)

    isotropic_points = np.array([(zi, yi, xi) for zi, yi, xi in zip(z, y, x)])
    # points are expressed as a ratio of the x dimension
    points = isotropic_points * truth_space_shape / truth_space_shape[2]

    return [(zi, yi, xi) for zi, yi, xi in points]


def _render_params_default_factory(data: dict[str, Any]) -> RenderingParams:
    path_points = _render_particle_path_points_factory(data)
    return RenderingParams(particle_path_points=path_points)


def _kymograph_sample_path_points_default_factory(
    data: dict[str, Any]
) -> list[tuple[float, float, float]]:
    assert isinstance(data["rendering"], RenderingParams)
    assert isinstance(data["imaging"], ImagingParams)

    truth_space_shape = np.array(data["imaging"].truth_space.shape)
    z_dims = truth_space_shape[0]
    z_mid = z_dims / 2
    # relative path points are expressed as a ratio of the x dimension
    relative_particle_path_points: NDArray[np.float_] = np.array(
        data["rendering"].particle_path_points, dtype=float
    )
    # path_points in pixel index space coordinates
    particle_path_points: NDArray[np.float_] = (
        relative_particle_path_points * truth_space_shape[2]
    )
    sample_path_points = particle_path_points.copy()
    # replace z with the mid point of the stack for sampling
    sample_path_points[:, 0] = z_mid
    return [(zi, yi, xi) for zi, yi, xi in sample_path_points]


def _kymograph_params_default_factory(data: dict[str, Any]) -> KymographParams:
    sample_path_points = _kymograph_sample_path_points_default_factory(data)
    return KymographParams(sample_path_points=sample_path_points)


class Params(BaseModel):

    model_config = ConfigDict(validate_assignment=True, validate_default=True)

    dynamics: DynamicsParams = DynamicsParams()

    imaging: "ImagingParams" = ImagingParams(
        truth_space=ms.ShapeScaleSpace(shape=(32, 64, 512), scale=(0.04, 0.01, 0.01)),
        output_space={"downscale": 4},
        modality=ms.Widefield(),
        detector=ms.CameraCCD(qe=1, read_noise=4, bit_depth=12, offset=100),
        exposure_ms=100,
    )

    rendering: RenderingParams = Field(default_factory=_render_params_default_factory)

    kymograph: KymographParams = Field(
        default_factory=_kymograph_params_default_factory
    )
