from typing import Any

import numpy as np
from numpy.typing import NDArray
from pydantic import ConfigDict, BaseModel, Field, create_model, model_validator
import microsim.schema as ms

from . import RenderParams, DynamicsParams, KymographParams


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


def _render_params_default_factory(data: dict[str, Any]) -> RenderParams:
    path_points = _render_particle_path_points_factory(data)
    return RenderParams(particle_path_points=path_points)


def _kymograph_sample_path_points_default_factory(
    data: dict[str, Any]
) -> list[tuple[float, float, float]]:
    assert isinstance(data["render"], RenderParams)
    assert isinstance(data["imaging"], ImagingParams)

    truth_space_shape = np.array(data["imaging"].truth_space.shape)
    z_dims = truth_space_shape[0]
    z_mid = z_dims / 2
    # relative path points are expressed as a ratio of the x dimension
    relative_particle_path_points: NDArray[np.float_] = np.array(
        data["render"].particle_path_points, dtype=float
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

    render: RenderParams = Field(default_factory=_render_params_default_factory)

    kymograph: KymographParams = Field(
        default_factory=_kymograph_params_default_factory
    )
