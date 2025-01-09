from typing import Any

import numpy as np
from numpy.typing import NDArray
from pydantic import ConfigDict, BaseModel, Field, create_model, model_validator
import microsim.schema as ms

from . import RenderingParams, DynamicsParams, KymographParams
from ..render.static_path import PiecewiseQuadraticBezierPath

# for type hinting
class ImagingParams(BaseModel): ...


def _build_imaging_params_class():
    # making a copy of microsim Simulation params without "sample" field
    _imaging_params_fields_dict = {}
    for field_name, field_info in ms.Simulation.model_fields.items():
        _imaging_params_fields_dict[field_name] = (field_info.annotation, field_info)
    del _imaging_params_fields_dict["sample"]
    return create_model(
        "ImagingParams",
        # adding validator like this is hacky... but there is only one so ok?
        # this is needed because it actually alters output space
        __validators__={
            "_resolve_spaces": model_validator(mode="after")(
                ms.Simulation._resolve_spaces
            )
        },
        **_imaging_params_fields_dict,
    )


ImagingParams = _build_imaging_params_class()

def _render_particle_path_points_factory(
    data: dict[str, Any]
) -> list[tuple[float, float, float]]:
    # truth space scale is in um
    truth_space_scale = np.array(data["imaging"].truth_space.scale)
    truth_space_shape = np.array(data["imaging"].truth_space.shape)
    desired_path_length = data["dynamics"].path_length

    n_points = 4
    x = np.zeros(n_points, dtype=float)
    x[0] = 0.1
    x[1:-1] = np.sort(np.random.uniform(0.2, 0.8, n_points - 2))
    x[-1] = 0.9
    y = np.random.uniform(0.1, 0.9, n_points)
    z = np.random.uniform(0.1, 0.9, n_points)

    isotropic_points = np.array([(zi, yi, xi) for zi, yi, xi in zip(z, y, x)])
    # now points are in um uints
    points = isotropic_points * truth_space_scale * truth_space_shape

    # TODO: I don't like this here maybe isotropic path points should be kept in params
    # scale points so that the path has the desired length
    initial_path = PiecewiseQuadraticBezierPath(points)
    new_points = points * desired_path_length / initial_path.length()

    return [(zi, yi, xi) for zi, yi, xi in new_points]


def _render_params_default_factory(data: dict[str, Any]) -> RenderingParams:
    path_points = _render_particle_path_points_factory(data)
    return RenderingParams(particle_path_points=path_points)


def _kymograph_sample_path_points_default_factory(
    data: dict[str, Any]
) -> list[tuple[float, float, float]]:
    assert isinstance(data["rendering"], RenderingParams)
    assert isinstance(data["imaging"], ImagingParams)

    truth_space_shape = np.array(data["imaging"].truth_space.shape)
    truth_space_scale = np.array(data["imaging"].truth_space.scale)
    downscale = data["imaging"].output_space.downscale
    z_dims = truth_space_shape[0]
    z_mid = z_dims / 2 / downscale 

    real_path_points = np.array(
        data["rendering"].particle_path_points, dtype=float
    )
    index_path_points = (real_path_points / np.array(truth_space_scale)) / downscale
    sample_path_points = index_path_points.copy()
    sample_path_points[:, 0] = z_mid
    return [(zi, yi, xi) for zi, yi, xi in sample_path_points]


def _kymograph_params_default_factory(data: dict[str, Any]) -> KymographParams:
    sample_path_points = _kymograph_sample_path_points_default_factory(data)
    return KymographParams(sample_path_points=sample_path_points)

def _imaging_params_default_factory(data: dict[str, Any]) -> ImagingParams:
    # path_points = 
    # TODO: choose rendering size from path coords
    # - have to swap rendering and imaging, another reason to calc shape in rendering

    return ImagingParams(
        truth_space=ms.ShapeScaleSpace(shape=(32, 64, 512), scale=(0.16, 0.04, 0.04)),
        output_space={"downscale": 4},
        modality=ms.Widefield(),
        detector=ms.CameraCCD(qe=1, read_noise=4, bit_depth=12, offset=100),
        exposure_ms=100,
    )

class Params(BaseModel):

    model_config = ConfigDict(validate_assignment=True, validate_default=True)

    dynamics: DynamicsParams = DynamicsParams()

    imaging: "ImagingParams" = Field(default_factory=_imaging_params_default_factory)

    rendering: RenderingParams = Field(default_factory=_render_params_default_factory)

    kymograph: KymographParams = Field(
        default_factory=_kymograph_params_default_factory
    )
