# Note: to run this script first us `uv sync --group cuda`
import numpy as np
import matplotlib.pyplot as plt
from kymograph_synthesis.pipeline import Pipeline
from kymograph_synthesis.params import Params
from kymograph_synthesis.pipeline.steps.generate_ground_truth import (
    generate_ground_truth,
    GroundTruthFuncCollection,
)
from kymograph_synthesis.params.render_params.render_params import (
    _random_relative_particle_path_points,
    _convert_relative_to_um,
)
from kymograph_synthesis.params.ground_truth_params import InstanceGroundTruth


def create_instance_map(instance_gt):
    instance_map = np.zeros(instance_gt.shape[:2])
    for i in range(instance_gt.shape[2]):
        instance_map[instance_gt[..., i] != 0] = i
    return instance_map


truth_shape = (32, 32, 768)
truth_scale = (0.04, 0.02, 0.02)
params = Params.model_validate(
    {
        "n_steps": 140,
        "dynamics": [
            {
                "particle_behaviour": "unidirectional",
                "particle_density": 2,
                "antero_speed_mode": 2,
                "antero_speed_var": 0.1**2,
                "retro_speed_mode": 1.5,
                "retro_speed_var": 0.1**2,
                "antero_resample_prob": 0.05,
                "retro_resample_prob": 0,
                "velocity_noise_var": 0.05**2,
                "fluorophore_count_mode": 400,
                "fluorophore_count_var": 40**2,
                "fluorophore_halflife_mode": 140,
                "fluorophore_halflife_var": 14**2,
                "state_ratios": {
                    "anterograde": 1,
                    "stationary": 0,
                    "retrograde": 0,
                },
                "seed": 42,
            },
            {
                "particle_behaviour": "unidirectional",
                "particle_density": 2,
                "antero_speed_mode": 2,
                "antero_speed_var": 0.1**2,
                "retro_speed_mode": 1.5,
                "retro_speed_var": 0.1**2,
                "antero_resample_prob": 0.05,
                "retro_resample_prob": 0,
                "velocity_noise_var": 0.05**2,
                "fluorophore_count_mode": 200,
                "fluorophore_count_var": 20**2,
                "fluorophore_halflife_mode": 140,
                "fluorophore_halflife_var": 14**2,
                "state_ratios": {
                    "anterograde": 0,
                    "stationary": 0,
                    "retrograde": 1,
                },
                "seed": 42,
            },
        ],
        "rendering": {
            "static_distributions": [
                {
                    "name": "simplex_noise",
                    "max_fluorophore_count_per_nm3": 0.01,
                    "noise_scales": [
                        0.5,
                        1.5,
                    ],
                    "scale_weights": [0.5, 0.5],
                    "seed": 42,
                }
            ],
            "particle_path_points": _convert_relative_to_um(
                _random_relative_particle_path_points(5),
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
                "objective_lens": {"na": 1.4},
                "settings": {"random_seed": 42, "np_backend": "cupy"},
            },
        },
        "kymograph": {
            "n_spatial_values_factor": 1,
            "projection_method": "centre_slice",
        },
        "ground_truth_funcs": [{"name": "instance", "project_to_xy": True}],
    }
)

pipeline = Pipeline(params, out_dir=None)
pipeline.run()


slice_proj_kymo_data = pipeline.sample_kymograph_output
slice_proj_kymo = slice_proj_kymo_data["kymograph"]
frames = pipeline.imaging_sim_output["frames"]
pipeline.params.kymograph.projection_method = "mean"
pipeline.run_sample_kymograph()
mean_proj_kymo = pipeline.sample_kymograph_output["kymograph"]
pipeline.params.kymograph.projection_method = "max"
pipeline.run_sample_kymograph()
max_proj_kymo = pipeline.sample_kymograph_output["kymograph"]
slice_proj_kymo = slice_proj_kymo_data["kymograph"]

instance_gt = generate_ground_truth(
    [InstanceGroundTruth(project_to_xy=True)],
    pipeline.dynamics_sim_output,
    slice_proj_kymo_data["n_spatial_values"],
    params.rendering.particle_path_points,
)[GroundTruthFuncCollection.INSTANCE]
instance_map = create_instance_map(instance_gt)

fig, axes = plt.subplots(1, 4, figsize=(15, 15))
axes[0].imshow(mean_proj_kymo, "gray", interpolation="nearest")
axes[0].set_title("Projection method: Mean")
axes[1].imshow(max_proj_kymo, "gray", interpolation="nearest")
axes[1].set_title("Projection method: Max")
axes[2].imshow(slice_proj_kymo, "gray", interpolation="nearest")
axes[2].set_title("Projection method: Centre Slice")
axes[3].imshow(instance_map, "turbo", interpolation="nearest")
axes[3].set_title("GT")
plt.show()
