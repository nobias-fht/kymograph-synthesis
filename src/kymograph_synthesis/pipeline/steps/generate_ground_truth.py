from typing import TypedDict, TypeAlias
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt

from ...params.ground_truth_params import (
    GroundTruthFuncParams,
    GroundTruthFuncCollection,
)
from ...ground_truth import generate_instance_ground_truth, generate_state_ground_truth
from ..utils import _resize_image
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


def save_ground_truth_visualization(
    generated_ground_truth: GenerateGroundTruthOutput,
    kymograph_gt_visual_fname: str,
    output_dir: Path,
):
    # TODO: validate that kymograph_gt_visual_fname contains {type} for formatting
    for key, kymograph_gt_raw in generated_ground_truth.items():
        file_path = output_dir / kymograph_gt_visual_fname.format(type=key.value)
        kymograph_gt_resized = _resize_image(kymograph_gt_raw, factor=4)
        match key:
            case GroundTruthFuncCollection.STATE:
                plt.imsave(
                    file_path,
                    kymograph_gt_resized * 255,
                )
            case GroundTruthFuncCollection.INSTANCE:
                instance_map = _create_instance_map(kymograph_gt_resized)
                plt.imsave(
                    file_path,
                    instance_map,
                    cmap="turbo",
                )
            case _:
                raise ValueError(
                    "Unrecognized ground truth generation function: " f"'{key}'."
                )


def _create_instance_map(instance_gt: NDArray) -> NDArray:
    instance_map = np.zeros(instance_gt.shape[:2])
    for i in range(instance_gt.shape[2]):
        instance_map[instance_gt[..., i] != 0] = i
    return instance_map
