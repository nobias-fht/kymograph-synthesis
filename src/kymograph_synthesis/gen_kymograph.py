from typing import Optional

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


def gen_kymograph_gt(positions: NDArray, path_range: tuple[float, float], ):
    ...