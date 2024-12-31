from typing import Literal, Optional

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import interp1d

from .render.static_path import StaticPath


def sample_kymograph(
    image_stack: NDArray,
    sample_path: StaticPath,
    path_clip: tuple[float, float],
    n_spatial_samples: int,
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
    ],
) -> NDArray:
    """
    Parameters
    ----------
    image_stack : numpy.ndarray
        Has shape T(Z)YX.
    sample_path : StaticPath
        Path along which samples are taken to create a kympgraph.
    n_spatial_samples : int
        Number of spatial samples.
    """
    n_time_samples = image_stack.shape[0]
    n_path_units = np.floor(sample_path.length())
    path_samples = np.linspace(0, 1, n_path_units)
    coords = sample_path(path_samples)
    indices = np.round(coords).astype(int)

    # place holder
    kymograph = np.zeros((n_time_samples, n_spatial_samples))

    # sample
    for t in range(n_time_samples):
        sample = image_stack[t, *[indices[:, i] for i in range(indices.shape[1])]]
        if interpolation is not None:
            sample = inter_pixel_interp(
                path_samples,
                indices,
                sample,
                interpolation=interpolation,
                new_path_samples=np.linspace(*path_clip, n_spatial_samples),
            )
        kymograph[t] = sample


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
    return interp_f(new_path_samples)
