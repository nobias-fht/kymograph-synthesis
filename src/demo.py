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
from kymograph_synthesis.render.static_path import LinearPath, QuadraticBezierPath
from kymograph_synthesis.render.fluorophore_distributions import (
    ParticleSystem,
    SimplexNoise,
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

intensity_mode = 300
intensity_var = 50**2

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
z_dim = 32
downscale = 4
# path_start = np.array([0.5, 0.1, 0.1])
# path_end = np.array([0.5, 0.9, 0.9])
# static_path = LinearPath(start=path_start, end=path_end)
# path_points = [
#     np.array([0.5, 0.5, 0.1]),
#     np.array([0.1, 0.9, 0.3]),
#     np.array([0.5, 0.5, 0.9])
# ]
# static_path = QuadraticBezierPath(points=path_points)

path_points = [
    np.array([0.1, 0.5, 0.1]),
    np.array([0.9, 0.5, 0.9]),
]
static_path = LinearPath(start=path_points[0], end=path_points[1])

# --- run through microsim
ground_truth_shape = (z_dim, 64, 512)
digital_simulation_frames: list[NDArray] = []
ground_truth_frames: list[NDArray] = []
static_distributions = [
    SimplexNoise(
        scales=[5, 10], scale_weights=[1, 1], max_intensity=intensity_mode * 10e-5
    )
]
for t in range(n_steps):
    time_varying_distributions = [
        ParticleSystem.on_static_path(
            static_path, particle_positions[t], particle_intensities[t]
        ),
    ]
    sim = ms.Simulation(
        truth_space=ms.ShapeScaleSpace(
            shape=ground_truth_shape, scale=(0.02, 0.01, 0.01)
        ),
        output_space={"downscale": downscale},
        sample=ms.Sample(labels=static_distributions + time_varying_distributions),
        modality=ms.Widefield(),
        detector=ms.CameraCCD(qe=1, read_noise=4, bit_depth=12, offset=100),
    )

    ground_truth = sim.ground_truth()
    digital_image = sim.digital_image(exposure_ms=200, with_detector_noise=True)
    ground_truth_frames.append(ground_truth)
    digital_simulation_frames.append(digital_image)
ground_truth = np.concatenate(ground_truth_frames)
digital_simulation = np.concatenate(digital_simulation_frames)


# --- create kymograph
spatial_samples = np.linspace(0.1, 0.9, n_spatial_samples)
kymograph = np.zeros((n_steps, n_spatial_samples))
# linear_path = LinearPath(
#     start=np.array(path_start) * ground_truth_shape / downscale, end=np.array(path_end) * ground_truth_shape / downscale
# )
kymograph_path_points = [
    point * ground_truth_shape / downscale for point in path_points
]
z_point = z_dim / 2 / downscale
for point in kymograph_path_points:
    point[0] = z_point
# kymo_sample_path = QuadraticBezierPath(
#     points=kymograph_path_points
# )
kymo_sample_path = LinearPath(
    start=kymograph_path_points[0], end=kymograph_path_points[1]
)
for t in range(n_steps):
    spatial_locations = kymo_sample_path(spatial_samples)
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
    digital_simulation[0, np.round(z_dim / 2 / downscale).astype(int)],
    cmap="gray",
    interpolation="none",
    vmin=digital_simulation.min(),
    vmax=digital_simulation.max(),
)


def update_digital_animation(frame):
    digital_animation_img.set_array(
        digital_simulation[frame, np.round(z_dim / 2 / downscale).astype(int)]
    )
    return digital_animation_img


digital_animation = FuncAnimation(
    fig, update_digital_animation, frames=n_steps, interval=200, repeat=True
)
digital_animation_ax.set_title("Digital particle simulation")

# --- ground truth animation
ground_truth_animation_img = ground_truth_animation_ax.imshow(
    ground_truth[0].sum(axis=0),
    cmap="gray",
    interpolation="none",
    vmin=digital_simulation.min(),
    vmax=digital_simulation.max(),
)


def update_ground_truth_animation(frame):
    ground_truth_animation_img.set_array(ground_truth[frame].sum(axis=0))
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
    kymograph_gt_ax.plot(
        particle_positions[:, i], np.arange(particle_positions.shape[0])
    )
kymograph_gt_ax.invert_yaxis()
kymograph_gt_ax.set_aspect(1 / kymograph.shape[1])
kymograph_gt_ax.set_ylabel("Time")
kymograph_gt_ax.set_xlabel("Position")
kymograph_gt_ax.set_xlim(0.1, 0.9)
kymograph_gt_ax.set_ylim(n_steps, 0)
kymograph_gt_ax.set_title("Kymograph ground truth")


# fig.tight_layout()
plt.show()
