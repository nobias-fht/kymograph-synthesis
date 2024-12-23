from typing import Literal

import numpy as np
from numpy.typing import NDArray
import microsim.schema as ms

from .render.static_path import PiecewiseQuadraticBezierPath
from .render.fluorophore_distributions import ParticleSystem


def gen_kymograph(
    particle_positions: NDArray,
    particle_intensities: NDArray,
    path_points: list[NDArray],
    ground_truth_shape: tuple[int, int, int],
    ground_truth_scale: tuple[float, float, float],
    downscale_factor: int,
    exposure_ms: float,
    static_background_distributions: list[ms.FluorophoreDistribution],
    # TODO: time varying distributions?
    # TODO: select camera and camera args
    # TODO: select modality and modality args
):
    static_path = PiecewiseQuadraticBezierPath(path_points)

    n_steps = particle_positions.shape[0]
    digital_sim = np.zeros(
        (n_steps, *[dim // downscale_factor for dim in ground_truth_shape])
    )
    for t in range(n_steps):
        particle_distribution = ParticleSystem.on_static_path(
            static_path, particle_positions[t], particle_intensities[t]
        )

        psf_modality = ms.Widefield()
        detector = ms.CameraCCD(qe=1, read_noise=4, bit_depth=12, offset=100)

        sim = ms.Simulation(
            truth_space=ms.ShapeScaleSpace(
                shape=ground_truth_shape, scale=ground_truth_scale
            ),
            output_path={"downscale": downscale_factor},
            sample=ms.Sample(
                labels=[particle_distribution, *static_background_distributions]
            ),
            modality=psf_modality,
            detector=detector,
        )
        digital_sim[t] = sim.digital_image(exposure_ms=200, with_detector_noise=True)

    # return kymograph and ground truth?
    # split into components? retrograde, anterograde, stationary
    # return 2D image ?


def gen_kymograph_gt(
    positions: NDArray, spatial_res: int, path_range: tuple[float, float] = (0, 1)
) -> NDArray:
    n_steps = positions.shape[0]
    n_spatial_samples = spatial_res

    # placeholder
    kymograph_gt = np.zeros((n_steps, n_spatial_samples))

    positions_scaled = (positions - path_range[0]) / (path_range[1] - path_range[0])
    midpoints = positions_scaled[:-1] + np.diff(positions_scaled, axis=0) / 2
    for t in range(n_steps):
        if t != 0:
            indices = _calc_interp_indices(
                spatial_res=spatial_res,
                positions_0=midpoints[t - 1],
                positions_1=positions_scaled[t],
            )
            intensities = _calc_interp_intensities(
                interp_indices=indices, direction="left"
            )
            indices = indices.flatten()
            intensities = intensities.flatten()
            in_bounds = (0 <= indices) & (indices < n_spatial_samples)
            kymograph_gt[t, indices[in_bounds]] = intensities[in_bounds]
        if t != n_steps - 1:
            indices = _calc_interp_indices(
                spatial_res=spatial_res,
                positions_0=positions_scaled[t],
                positions_1=midpoints[t],
            )
            intensities = _calc_interp_intensities(
                interp_indices=indices, direction="right"
            )
            indices = indices.flatten()
            intensities = intensities.flatten()
            in_bounds = (0 <= indices) & (indices < n_spatial_samples)
            kymograph_gt[t, indices[in_bounds]] = intensities[in_bounds]
    return kymograph_gt


def _calc_interp_indices(spatial_res: int, positions_0: NDArray, positions_1: NDArray):
    gradient = positions_1 - positions_0
    index_gradient = np.abs(gradient * spatial_res)
    index_gradient = index_gradient[~np.isnan(index_gradient)]
    if len(index_gradient) == 0:
        interp_n = 1
    else:
        interp_n = int(np.ceil(index_gradient).max()) + 1
    interp_positions = np.linspace(positions_0, positions_1, interp_n)
    interp_indices = np.round(interp_positions * spatial_res).astype(int)
    return interp_indices


def _calc_interp_intensities(
    interp_indices: NDArray[np.int_],
    direction: Literal["left", "right"],
):
    match direction:
        case "left":
            original_indices = interp_indices[[-1]]
        case "right":
            original_indices = interp_indices[[0]]
        case _:
            raise ValueError(f"Unknown value for directions '{direction}'.")

    interp_n = interp_indices.shape[0]
    intensities = 1 - abs(original_indices - interp_indices) / interp_n
    return intensities
