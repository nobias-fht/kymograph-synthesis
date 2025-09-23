from typing import Optional, TypedDict

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import map_coordinates

from ...params import KymographParams
from ...render.static_path import PiecewiseQuadraticBezierPath


class SampleKymographOutput(TypedDict):
    kymograph: NDArray
    path_length_pixels: float
    n_spatial_values: int


# TODO: output ndarray type, see if interp changes it.
def sample_kymograph(
    params: KymographParams,
    frames: NDArray[np.uint16],
) -> SampleKymographOutput:
    # - 0.5 for map_coordinates (centre vs edge of pixel)
    sample_path_points = [np.array(point) - 0.5 for point in params.sample_path_points]
    sample_path = PiecewiseQuadraticBezierPath(sample_path_points)

    n_spatial_values = int(
        np.round(sample_path.length() * params.n_spatial_values_factor)
    )
    n_time_samples = frames.shape[0]
    path_samples = np.linspace(0, 1, n_spatial_values)

    # place holder
    kymograph = np.zeros((n_time_samples, n_spatial_values), dtype=frames.dtype)
    coords = sample_path(path_samples).squeeze()
    for t in range(n_time_samples):
        kymograph[t] = map_coordinates(
            frames[t, 3], coords[:, 1:].T, order=params.interpolation
        )

    return SampleKymographOutput(
        kymograph=kymograph,
        path_length_pixels=int(np.floor(sample_path.length())),
        n_spatial_values=n_spatial_values,
    )
