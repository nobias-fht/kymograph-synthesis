from typing import TypedDict, TypeAlias
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from ...params.ground_truth_params import (
    GroundTruthFuncParams,
    GroundTruthFuncCollection,
)
from ...ground_truth import generate_instance_ground_truth, generate_state_ground_truth
from .simulate_dynamics import DynamicsSimOutput

GenerateGroundTruthOutput: TypeAlias = dict[GroundTruthFuncCollection, NDArray]


def generate_ground_truth(
    params: Sequence[GroundTruthFuncParams],
    dynamics_output: DynamicsSimOutput,
    n_spatial_values: int,
):
    output: GenerateGroundTruthOutput = {}
    for ground_truth_func_params in params:
        # TODO: make a nicer switching pattern?
        match ground_truth_func_params.name:
            case GroundTruthFuncCollection.STATE:
                output[GroundTruthFuncCollection.STATE] = generate_state_ground_truth(
                    dynamics_output["particle_positions"],
                    dynamics_output["particle_states"],
                    n_spatial_values,
                )
            case GroundTruthFuncCollection.INSTANCE:
                output[GroundTruthFuncCollection.INSTANCE] = (
                    generate_instance_ground_truth(
                        dynamics_output["particle_positions"], n_spatial_values
                    )
                )
            case _:
                raise ValueError(
                    "Unrecognized ground truth generation function: "
                    f"'{ground_truth_func_params.name}'."
                )
    return output
