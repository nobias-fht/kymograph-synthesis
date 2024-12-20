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
from kymograph_synthesis.render.static_path import (
    LinearPath,
    QuadraticBezierPath,
    PiecewiseQuadraticBezierPath,
)
from kymograph_synthesis.render.fluorophore_distributions import (
    ParticleSystem,
    SimplexNoise,
)
from kymograph_synthesis.sample_kymograph import inter_pixel_interp

# --- simulation params
n_steps = 64
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

transition_matrix = {
    MotionStateCollection.ANTEROGRADE: {
        MotionStateCollection.ANTEROGRADE: 0.9,
        MotionStateCollection.STATIONARY: 0.1,
        MotionStateCollection.RETROGRADE: 0,
    },
    MotionStateCollection.STATIONARY: {
        MotionStateCollection.ANTEROGRADE: 0.5,
        MotionStateCollection.STATIONARY: 0.5,
        MotionStateCollection.RETROGRADE: 0,
    },
    MotionStateCollection.RETROGRADE: {
        MotionStateCollection.ANTEROGRADE: 0,
        MotionStateCollection.STATIONARY: 0,
        MotionStateCollection.RETROGRADE: 1,
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
    intensity_half_life_mode=n_steps * 2,
    intensity_half_life_var=n_steps / 2,
    velocity_noise_std=velocity_noise_std,
    transition_matrix=transition_matrix,
    n_steps=n_steps,
)
particle_positions, particle_intensities, particle_states = run_simulation(
    n_steps, particles
)

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
    np.array([0.5, 0.5, 0.1]),
    np.array([0.5, 0.5, 0.6]),
    np.array([0.8, 0.8, 0.7]),
    np.array([0.5, 0.5, 0.8]),
    np.array([0.5, 0.5, 0.9]),
]

# --- run through microsim
ground_truth_shape = (z_dim, 64, 512)
static_path = PiecewiseQuadraticBezierPath(
    points=[
        point * np.array(ground_truth_shape) / max(ground_truth_shape)
        for point in path_points
    ]
)
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
            shape=ground_truth_shape, scale=(0.04, 0.01, 0.01)
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

# --- ground truth kymo
particle_positions_scaled = (particle_positions - 0.1)/(0.9 - 0.1)

midpoints = particle_positions_scaled[:-1] + np.diff(particle_positions_scaled, axis=0)/2

kymograph_gt = np.zeros((n_steps, n_spatial_samples))
for t in range(n_steps):
    original_indices = np.round(particle_positions_scaled[t] * n_spatial_samples)
    if t != 0:
        # -- behind
        gradient = particle_positions_scaled[t] - midpoints[t - 1]
        interp_n = int(np.ceil(np.abs(gradient*n_spatial_samples).max())) + 1
        interp_positions = np.linspace(midpoints[t - 1], particle_positions_scaled[t], interp_n)
        for position in interp_positions:
            indices = np.round(position * n_spatial_samples).astype(int)
            intensities = 1 - abs(original_indices - indices)/interp_n
            in_bounds = (0 <= indices) & (indices < n_spatial_samples)
            indices = indices[in_bounds]
            # kymograph_gt[t, indices] += intensities[in_bounds]
            kymograph_gt[t, indices] = 1

    if t != n_steps - 1:
        # --- in front
        gradient = midpoints[t] - particle_positions_scaled[t]
        interp_n = int(np.ceil(np.abs(gradient*n_spatial_samples).max())) + 1
        interp_positions = np.linspace(particle_positions_scaled[t], midpoints[t], interp_n)
        for position in interp_positions:
            indices = np.round(position * n_spatial_samples).astype(int)
            intensities = 1 - abs(original_indices - indices)/interp_n
            in_bounds = (0 <= indices) & (indices < n_spatial_samples)
            indices = indices[in_bounds]
            # kymograph_gt[t, indices] += intensities[in_bounds]
            kymograph_gt[t, indices] = 1



# --- create kymograph
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
print("Kymo path points")
print(kymograph_path_points)
kymo_sample_path = PiecewiseQuadraticBezierPath(points=kymograph_path_points)
spatial_samples = np.linspace(0, 1, int(np.ceil(kymo_sample_path.length())))
for t in range(n_steps):
    spatial_locations = kymo_sample_path(spatial_samples)
    coords = np.round(spatial_locations).astype(int)
    time_sample = digital_simulation[t, coords[:, 0], coords[:, 1], coords[:, 2]]
    time_sample = inter_pixel_interp(
        spatial_samples, coords, time_sample, np.linspace(0.1, 0.9, n_spatial_samples)
    )
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

# # --- diplay kymograph ground truth
# for i in range(particle_positions.shape[1]):
#     kymograph_gt_ax.plot(
#         particle_positions[:, i], np.arange(particle_positions.shape[0])
#     )
# kymograph_gt_ax.invert_yaxis()
# kymograph_gt_ax.set_xlim(0.1, 0.9)
# kymograph_gt_ax.set_aspect((0.9 - 0.1) / kymograph.shape[1])
# kymograph_gt_ax.set_ylabel("Time")
# kymograph_gt_ax.set_xlabel("Position")
# kymograph_gt_ax.set_ylim(n_steps, 0)
# kymograph_gt_ax.set_title("Kymograph ground truth")
kymograph_gt_ax.imshow(kymograph_gt, cmap="gray")

# fig.tight_layout()
plt.show()
