import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec
from microsim import schema as ms
from microsim.util import ortho_plot

from kymograph_synthesis.dynamics.particle_simulator.motion_state_collection import (
    MotionStateCollection,
)
from kymograph_synthesis.dynamics.system_simulator import (
    create_particle_simulators,
    run_simulation,
)
from kymograph_synthesis.render.ground_truth import render
from kymograph_synthesis.render.static_path import LinearPath
from kymograph_synthesis.render.space_time_objects import (
    SpaceTimeObject,
    ParticleSystem,
    StaticSimplexNoise,
)

# --- simulation params
n_steps = 128
particle_density = 6

# retro_speed_mode = 0.2e-2
# retro_speed_var = 0.02e-2 ** 2
retro_speed_mode = 2.4e-2
retro_speed_var = 0.2e-2**2
antero_speed_mode = 1.6e-2
antero_speed_var = 0.1e-2**2
velocity_noise_std = 0.32e-2

intensity_mode = 600
intensity_var = 100 ** 2

n_spatial_samples = 96

switch_prob = {
    MotionStateCollection.ANTEROGRADE: 0.1,
    MotionStateCollection.STATIONARY: 0.5,
    MotionStateCollection.RETROGRADE: 0,
}
transition_matrix = {
    MotionStateCollection.ANTEROGRADE: {
        MotionStateCollection.ANTEROGRADE: 0,
        MotionStateCollection.STATIONARY: 1,
        MotionStateCollection.RETROGRADE: 0,
    },
    MotionStateCollection.STATIONARY: {
        MotionStateCollection.ANTEROGRADE: 1,
        MotionStateCollection.STATIONARY: 0,
        MotionStateCollection.RETROGRADE: 0,
    },
    MotionStateCollection.RETROGRADE: {
        MotionStateCollection.ANTEROGRADE: 0,
        MotionStateCollection.STATIONARY: 1,
        MotionStateCollection.RETROGRADE: 0,
    },
}

# --- generate particle properties from simulation
particles = create_particle_simulators(
    particle_density,
    antero_speed_mode,
    antero_speed_var,
    retro_speed_mode,
    retro_speed_var,
    intensity_mode=intensity_mode,
    intensity_var=intensity_var,
    intensity_half_life_mode=n_steps,
    intensity_half_life_var=n_steps / 4,
    velocity_noise_std=velocity_noise_std,
    state_switch_prob=switch_prob,
    transition_prob_matrix=transition_matrix,
    n_steps=n_steps,
)
particle_positions, particle_intensities = run_simulation(n_steps, particles)

# --- render ground truth
z_dim = 16
downscale = 4
path_start = np.array([z_dim // 2, 64 - 16, 64 - 16])
path_end = np.array([z_dim // 2, 16, 512 - 16])
static_path = LinearPath(start=path_start, end=path_end)
objects: list[SpaceTimeObject] = [
    ParticleSystem.on_static_path(
        static_path, particle_positions, particle_intensities
    ),
    StaticSimplexNoise(
        scales=[5, 10], scale_weights=[1, 1], max_intensity=intensity_mode * 10e-5
    ),
]
space_time_gt = np.zeros((n_steps, z_dim, 64, 512))
for object in objects:
    object.render(space_time=space_time_gt)

# --- run through microsim
digital_simulation_frames: list[NDArray] = []
for i, frame_gt in enumerate(space_time_gt):
    sim = ms.Simulation.from_ground_truth(
        frame_gt,
        scale=(0.2, 0.01, 0.01),
        output_space={"downscale": downscale},
        modality=ms.Widefield(),
        detector=ms.CameraCCD(qe=1, read_noise=4, bit_depth=12, offset=100),
    )
    digital_image = sim.digital_image(exposure_ms=300, with_detector_noise=True)
    digital_simulation_frames.append(digital_image)
digital_simulation = np.concatenate(digital_simulation_frames)

# --- create kymograph
spatial_samples = np.linspace(0, 1, n_spatial_samples)
kymograph = np.zeros((n_steps, n_spatial_samples))
linear_path = LinearPath(
    start=np.array(path_start) / downscale, end=np.array(path_end) / downscale
)
for t in range(n_steps):
    spatial_locations = linear_path(spatial_samples)
    coords = np.round(spatial_locations).astype(int)
    time_sample = digital_simulation[t, coords[:, 0], coords[:, 1], coords[:, 2]]
    kymograph[t] = time_sample

# ---- display
fig = plt.figure(layout="constrained", figsize=(16, 6))
gs = GridSpec(2, 3, figure=fig, width_ratios=[2, 1.05, 1.05])
digital_animation_ax = fig.add_subplot(gs[0, 0])
ground_truth_animation_ax = fig.add_subplot(gs[1, 0])
kymograph_ax = fig.add_subplot(gs[:, 1])
kymograph_gt_ax = fig.add_subplot(gs[:, 2])

# --- digital sim animation
digital_animation_img = digital_animation_ax.imshow(
    digital_simulation[0, z_dim // 2 // downscale],
    cmap="gray",
    interpolation="none",
    vmin=digital_simulation.min(),
    vmax=digital_simulation.max(),
)


def update_digital_animation(frame):
    digital_animation_img.set_array(digital_simulation[frame, z_dim // 2 // downscale])
    return digital_animation_img


digital_animation = FuncAnimation(
    fig, update_digital_animation, frames=n_steps, interval=200, repeat=True
)
digital_animation_ax.set_title("Digital particle simulation")

# --- ground truth animation
ground_truth_animation_img = ground_truth_animation_ax.imshow(
    space_time_gt[0, z_dim // 2],
    cmap="gray",
    interpolation="none",
    vmin=digital_simulation.min(),
    vmax=digital_simulation.max(),
)


def update_ground_truth_animation(frame):
    ground_truth_animation_img.set_array(space_time_gt[frame, z_dim // 2])
    return ground_truth_animation_img


ground_truth_animation = FuncAnimation(
    fig, update_ground_truth_animation, frames=n_steps, interval=200, repeat=True
)
ground_truth_animation_ax.set_title("Particle simulation")

# --- display kymograph
kymograph_ax.imshow(kymograph, cmap="gray", interpolation="none")
kymograph_ax.set_ylabel("Time")
kymograph_ax.set_xlabel("Position")
kymograph_ax.set_title("Kymograph")

# --- diplay kymograph ground truth
for i in range(particle_positions.shape[1]):
    kymograph_gt_ax.plot(particle_positions[:, i], np.arange(particle_positions.shape[0]))
kymograph_gt_ax.invert_yaxis()
kymograph_gt_ax.set_aspect(1/kymograph.shape[1])
kymograph_gt_ax.set_ylabel("Time")
kymograph_gt_ax.set_xlabel("Position")
kymograph_gt_ax.set_xlim(0, 1)
kymograph_gt_ax.set_ylim(n_steps, 0)
kymograph_gt_ax.set_title("Kymograph ground truth")


# fig.tight_layout()
plt.show()