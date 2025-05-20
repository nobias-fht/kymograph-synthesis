from typing import Optional, TypedDict

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import interp1d

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
    sample_path_points = [np.array(point) for point in params.sample_path_points]
    sample_path = PiecewiseQuadraticBezierPath(sample_path_points)

    n_time_samples = frames.shape[0]
    path_length_pixels = sample_path.length()
    n_path_units = int(np.floor(path_length_pixels))
    path_samples = np.linspace(0, 1, n_path_units)
    coords = sample_path(path_samples)
    indices = np.round(coords).astype(int)

    if params.interpolation != "none":
        n_spatial_values = int(np.round(n_path_units * params.n_spatial_values_factor))
    else:
        n_spatial_values = n_path_units
    # place holder
    kymograph = np.zeros((n_time_samples, n_spatial_values), dtype=frames.dtype)

    # sample
    for t in range(n_time_samples):
        sample = frames[t, *[indices[:, i] for i in range(indices.shape[1])]]
        if params.interpolation != "none":
            sample = inter_pixel_interp(
                path_samples,
                indices,
                sample,
                interpolation=params.interpolation,
                new_path_samples=np.linspace(0, 1, n_spatial_values),
            )
        kymograph[t] = sample
    return SampleKymographOutput(
        kymograph=kymograph,
        path_length_pixels=path_length_pixels,
        n_spatial_values=n_spatial_values,
    )


def inter_pixel_interp(
    path_samples: NDArray,
    pixel_indices: NDArray,
    values: NDArray,
    new_path_samples: Optional[NDArray],
    interpolation="cubic",
):
    if new_path_samples is None:
        new_path_samples = path_samples
    # TODO: explain
    unique, unique_indices = np.unique(pixel_indices, axis=0, return_index=True)

    # placeholder
    n_unique = len(unique)
    average_path_samples = np.zeros(n_unique)
    for i, val in enumerate(unique):
        average_path_samples[i] = np.mean(
            path_samples[(pixel_indices == val).all(axis=1)]
        )

    interp_f = interp1d(
        average_path_samples,
        values[unique_indices],
        kind=interpolation,
        fill_value="extrapolate",
    )
    return interp_f(new_path_samples).astype(path_samples.dtype)
