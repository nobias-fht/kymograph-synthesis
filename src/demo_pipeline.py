from pprint import pprint

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from kymograph_synthesis.params import Params
from kymograph_synthesis.pipeline import Pipeline
from kymograph_synthesis.dynamics.particle_simulator.motion_state_collection import (
    MotionStateCollection,
)


params = Params(
    dynamics={
        "seed": 42,
        # "n_steps": 64,
        "particle_density": 8,
        "fluorophore_count_mode": 400,
        "fluorophore_count_var": 100**2,
        "retro_speed_mode": 1.6,
        "retro_speed_var": 0.0001,
        "antero_speed_mode": 2,
        "antero_speed_var": 0.0001,
        "velocity_noise_var": 0.005,
        "particle_behaviour": "unidirectional",
        # "transition_matrix": {
        #     MotionStateCollection.ANTEROGRADE: {
        #         MotionStateCollection.ANTEROGRADE: 1,
        #         MotionStateCollection.STATIONARY: 0,
        #         MotionStateCollection.RETROGRADE: 0,
        #     },
        #     MotionStateCollection.STATIONARY: {
        #         MotionStateCollection.ANTEROGRADE: 0,
        #         MotionStateCollection.STATIONARY: 1,
        #         MotionStateCollection.RETROGRADE: 0,
        #     },
        #     MotionStateCollection.RETROGRADE: {
        #         MotionStateCollection.ANTEROGRADE: 0,
        #         MotionStateCollection.STATIONARY: 0,
        #         MotionStateCollection.RETROGRADE: 1,
        #     },
        # },
        "fluorophore_halflife_mode": 64,
        "fluorophore_halflife_var": 0.6,
    },
    rendering={
        # "particle_path_points": [
        #     (0.5, 0.9, 0.1),
        #     (0.5, 0.2, 0.7),
        #     (0.7, 0.9, 0.8),
        #     (0.8, 0.1, 0.9),
        # ],
        "static_distributions": [
            {
                "name": "simplex_noise",
                "max_fluorophore_count_per_nm3": 0.02,
                "noise_scales": [2, 0.5],
                "scale_weights": [1, 0.75],
                "seed": 42,
            }
        ],
        "imaging": {
            "exposure_ms": 100,
            "truth_space": {"shape": [24, 64, 512], "scale": [0.04, 0.02, 0.02]},
            "detector": {"camera_type": "CCD", "read_noise": 6},
            "settings": {"random_seed": 42},
        },
    },
    # kymograph={"interpolation": "none"}
)
pprint(params.model_dump(exclude={"rendering": {"imaging": ["channels"]}}))

pipeline = Pipeline(params)

pipeline.run()

digital_simulation = pipeline.imaging_sim_output.frames
digital_display_z_index = params.rendering.imaging.output_space.shape[0] // 2

digital_animation_fig, digital_animation_ax = plt.subplots()
digital_animation_img = digital_animation_ax.imshow(
    digital_simulation[0, digital_display_z_index],
    cmap="gray",
    interpolation="none",
    vmin=digital_simulation.min(),
    vmax=digital_simulation.max(),
)


def update_digital_animation(frame):
    digital_animation_img.set_array(digital_simulation[frame, digital_display_z_index])
    return digital_animation_img


digital_animation = FuncAnimation(
    digital_animation_fig,
    update_digital_animation,
    frames=params.dynamics.n_steps,
    interval=200,
    repeat=True,
)
digital_animation_ax.set_title("Digital particle simulation")

# --- kymograph
kymo_fig, kymo_ax = plt.subplots()
kymograph= pipeline.sample_kymograph_output.kymograph
kymo_ax.imshow(kymograph, cmap="gray")
kymo_ax.set_xlabel("Distance")
kymo_ax.set_ylabel("Time")
kymo_ax.set_title("Kymograph")
# particle_positions = pipeline.simulate_dynamics_output.particle_positions
# particle_pixel_positions = particle_positions * pipeline.sample_kymograph_output.n_spatial_values
# for p in range(particle_positions.shape[1]):
#     kymo_ax.plot(particle_pixel_positions[:, p]-0.5, np.arange(particle_positions.shape[0]), alpha=0.6)
# kymo_ax.set_xlim(-0.5, kymograph.shape[1]-0.5)
# kymo_ax.set_ylim(kymograph.shape[0]-0.5, -0.5)

# --- kymograph groundtruth
kymogt_fig, kymogt_ax = plt.subplots()
kymogt_ax.imshow(pipeline.generate_ground_truth_output.ground_truth.any(axis=-1), cmap="gray")
kymogt_ax.set_xlabel("Distance")
kymogt_ax.set_ylabel("Time")
kymogt_ax.set_title("Kymograph GT")
particle_positions = pipeline.dynamics_sim_output.particle_positions
particle_pixel_positions = particle_positions * pipeline.sample_kymograph_output.n_spatial_values
for p in range(particle_positions.shape[1]):
    kymogt_ax.plot(particle_pixel_positions[:, p]-0.5, np.arange(particle_positions.shape[0]), alpha=0.6)
kymogt_ax.set_xlim(-0.5, kymograph.shape[1]-0.5)
kymogt_ax.set_ylim(kymograph.shape[0]-0.5, -0.5)
plt.show()