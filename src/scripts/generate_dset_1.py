from typing import Optional
import argparse
from pathlib import Path

import numpy as np

from kymograph_synthesis.params import Params
from kymograph_synthesis.pipeline import Pipeline


def main(output_dir: Path, n_kymographs: int, seed: Optional[int]):

    rng = np.random.default_rng(seed=seed)

    for _ in range(n_kymographs):

        params = Params(
            dynamics={
                "particle_behaviour": "unidirectional",
                "particle_density": rng.uniform(1, 8),
                "antero_speed_mode": rng.uniform(1, 3),
                "antero_speed_var": rng.uniform(0.00005, 0.00015),
                "retro_speed_mode": rng.uniform(1, 3),
                "retro_speed_var": rng.uniform(0.00005, 0.00015),
                "velocity_noise_var": rng.uniform(0.01**2, 0.1**2),
                "fluorophore_count_mode": rng.uniform(200, 600),
                "fluorophore_count_var": rng.uniform(100**2, 300**2),
            },
            rendering={
                "static_distributions": [
                    {
                        "name": "simplex_noise",
                        "max_fluorophore_count_per_nm3": rng.uniform(0.005, 0.03),
                        "noise_scales": [rng.uniform(0.25, 0.75), rng.uniform(1, 1.5)],
                        "scale_weights": [rng.random(), rng.random()],
                    }
                ],
                "imaging": {
                    "exposure_ms": 100,
                    "truth_space": {
                        "shape": [24, 64, 512],
                        "scale": [0.04, 0.02, 0.02],
                    },
                    "detector": {
                        "camera_type": "CCD",
                        "read_noise": 6,
                        "gain": 60,
                        "offset": 600,
                    },
                },
            },
        )
        if seed is not None:
            params.dynamics.seed = seed
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
