from typing import Optional
import argparse
from pathlib import Path
from pprint import pprint

import numpy as np

from kymograph_synthesis.params import Params, DynamicsParams
from kymograph_synthesis.pipeline import Pipeline
from kymograph_synthesis.params.render_params.render_params import (
    _random_relative_particle_path_points,
    _convert_relative_to_um,
)


def main(output_dir: Path, n_kymographs: int, seed: Optional[int]):

    # TODO: make
    rng = np.random.default_rng(seed=seed)

    for _ in range(n_kymographs):
        n_steps = np.random.randint(96, 300)
        antero_speed_mode = rng.uniform(0.5, 3)
        antero_speed_var = (rng.uniform(0, 0.05) * antero_speed_mode) ** 2.5
        retro_speed_mode = rng.uniform(0.5, 3)
        retro_speed_var = (rng.uniform(0, 0.05) * retro_speed_mode) ** 2.5
        fluorophore_count_mode_set1 = rng.uniform(300, 600)
        fluorophore_count_mode_set2 = rng.uniform(300, 600)
        fluorophore_count_var_set1 = (
            rng.uniform(0, 0.3) * fluorophore_count_mode_set1
        ) ** 2
        fluorophore_count_var_set2 = (
            rng.uniform(0, 0.3) * fluorophore_count_mode_set2
        ) ** 2
        fluorophore_halflife_mode = rng.uniform(n_steps * 1, n_steps * 3)
        fluorophore_halflife_var = (
            rng.uniform(0, 0.1) * fluorophore_halflife_mode
        ) ** 2
        truth_shape = (32, 32, 768)
        x_scale = rng.choice([0.016, 0.018, 0.020, 0.022, 0.026, 0.028, 0.03, 0.032])
        truth_scale = (x_scale*2, x_scale, x_scale)

        params = Params.model_validate(
            {
                "n_steps": n_steps,
                "dynamics": [
                    {
                        "particle_behaviour": "unidirectional",
                        "particle_density": rng.uniform(0.25, 6),
                        "antero_speed_mode": antero_speed_mode,
                        "antero_speed_var": antero_speed_var,
                        "retro_speed_mode": retro_speed_mode,
                        "retro_speed_var": retro_speed_var,
                        "antero_resample_prob": rng.uniform(0, 0.1),
                        "retro_resample_prob": rng.uniform(0, 0.1),
                        "velocity_noise_var": rng.uniform(0.01**2, 0.05**2),
                        "fluorophore_count_mode": fluorophore_count_mode_set1,
                        "fluorophore_count_var": fluorophore_count_var_set1,
                        "fluorophore_halflife_mode": fluorophore_halflife_mode,
                        "fluorophore_halflife_var": fluorophore_halflife_var,
                        "state_ratios": {
                            "anterograde": 1,
                            "stationary": 0,
                            "retrograde": 0,
                        },
                    },
                    {
                        "particle_behaviour": "unidirectional",
                        "particle_density": rng.uniform(0.25, 6),
                        "antero_speed_mode": antero_speed_mode,
                        "antero_speed_var": antero_speed_var,
                        "retro_speed_mode": retro_speed_mode,
                        "retro_speed_var": retro_speed_var,
                        "antero_resample_prob": rng.uniform(0, 0.1),
                        "retro_resample_prob": rng.uniform(0, 0.1),
                        "velocity_noise_var": rng.uniform(0.01**2, 0.05**2),
                        "fluorophore_count_mode": fluorophore_count_mode_set2,
                        "fluorophore_count_var": fluorophore_count_var_set2,
                        "fluorophore_halflife_mode": fluorophore_halflife_mode,
                        "fluorophore_halflife_var": fluorophore_halflife_var,
                        "state_ratios": {
                            "anterograde": 0,
                            "stationary": 0,
                            "retrograde": 1,
                        },
                    },
                ],
                "rendering": {
                    "static_distributions": [
                        {
                            "name": "simplex_noise",
                            "max_fluorophore_count_per_nm3": rng.uniform(0.005, 0.03),
                            "noise_scales": [
                                rng.uniform(0.25, 0.75),
                                rng.uniform(1, 1.5),
                            ],
                            "scale_weights": [rng.random(), rng.random()],
                        }
                    ],
                    "particle_path_points": _convert_relative_to_um(
                        _random_relative_particle_path_points(rng.integers(4, 8)),
                        truth_shape,
                        truth_scale,
                    ),
                    "imaging": {
                        "exposure_ms": 100,
                        "truth_space": {
                            "shape": truth_shape,
                            "scale": truth_scale,
                        },
                        "detector": {
                            "camera_type": "CCD",
                            "read_noise": 6,
                            "gain": 10,
                            "offset": 100,
                        },
                        "objective_lens": {"na": rng.uniform(0.1, 1.5)},
                    },
                },
                "kymograph": {"n_spatial_values_factor": rng.uniform(0.8, 1.6)},
                "settings": {"np_backend": "cupy"}
            }
        )
        pprint(params.model_dump(mode="json"), indent=2)
        if seed is not None:
            if isinstance(params.dynamics, DynamicsParams):
                params.dynamics.seed = seed
            else:
                for param_set in params.dynamics:
                    param_set.seed = seed
            params.rendering.static_distributions[0].seed = seed
            params.rendering.imaging.settings.random_seed = seed
        pipeline = Pipeline(params, out_dir=output_dir)
        pipeline.run()
        pipeline.save()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dir", "-d", type=Path, required=True, help="Directory to save outputs to."
    )
    parser.add_argument(
        "-n",
        dest="n_kymographs",
        type=int,
        required=True,
        help="The number of kymographs to create.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        required=False,
        default=None,
        help="A seed for reproducibility",
    )
    args = parser.parse_args()

    main(output_dir=args.dir, n_kymographs=args.n_kymographs, seed=args.seed)
