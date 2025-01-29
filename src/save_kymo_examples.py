from pathlib import Path
import glob

import numpy as np

from kymograph_synthesis.params import Params
from kymograph_synthesis.pipeline import Pipeline


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


root_dir = Path("/Users/melisande.croft/Documents/Data/synthetic_kymos_2025-02-20")

existing = glob.glob(str(root_dir / f"kymo_[0-9][0-9][0-9].png"))
indices = [int(filename[-7:-4]) for filename in existing]
if len(indices) == 0:
    index = 0
else:
    index = max(indices) + 1

relative_path_points = [
    (0.5018493253329034, 0.1242217081708442, 0.2647286353826447),
    (0.43781181939192043, 0.508499636956678, 0.3653283497192502),
    (0.4258251739210646, 0.4873320288084215, 0.6369160672238992),
    (0.38297045657020434, 0.8423161047700577, 0.8767481449958608),
]
truth_space_shape = [24, 64, 512]
truth_space_scale = [0.04, 0.02, 0.02]
path_points = _convert_relative_to_um(
    relative_path_points, truth_space_shape, truth_space_scale
)

params = Params(
    dynamics={
        "seed": 42,
        "n_steps": 120,
        "particle_density": 8,
        "fluorophore_count_mode": 400,
        "fluorophore_count_var": 100**2,
        "retro_speed_mode": 1.6,
        "retro_speed_var": 0.0001,
        "antero_speed_mode": 2,
        "antero_speed_var": 0.0001,
        "velocity_noise_var": 0.005,
        "particle_behaviour": "unidirectional",
        "fluorophore_halflife_mode": 64,
        "fluorophore_halflife_var": 0.6,
    },
    rendering={
        "particle_path_points": path_points,
        "static_distributions": [
            {
                "name": "simplex_noise",
                "max_fluorophore_count_per_nm3": 0.01,
                "noise_scales": [0.5, 1],
                "scale_weights": [1, 0.75],
                "seed": 42,
            }
        ],
        "imaging": {
            "exposure_ms": 50,
            "truth_space": {"shape": truth_space_shape, "scale": truth_space_scale},
            "detector": {"camera_type": "CCD", "read_noise": 6},
            "settings": {"random_seed": 42},
            "objective_lens": {"numerical_aperture": 0.5}
        },
    },
)

pipeline = Pipeline(params)
pipeline.run()
pipeline.save(root_dir)
