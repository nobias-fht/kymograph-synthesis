from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class GenerateGroundTruthOutput:
    ground_truth: NDArray


def generate_ground_truth(
    particle_positions: NDArray[np.float_],
    particle_states: NDArray[np.int_],
    n_spatial_values: int,
): ...
