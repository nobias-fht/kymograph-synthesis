import numpy as np
import microsim.schema as ms
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from pprint import pprint

from kymograph_synthesis.params import Params
from kymograph_synthesis.dynamics.system_simulator import (
    create_particle_simulators,
    run_dynamics_simulation,
)
from kymograph_synthesis.render.static_path import PiecewiseQuadraticBezierPath
from kymograph_synthesis.render.fluorophore_distributions import (
    ParticleSystem,
    SimplexNoise,
)
from kymograph_synthesis.sample_kymograph import inter_pixel_interp
from kymograph_synthesis.dynamics.particle_simulator.motion_state_collection import (
    MotionStateCollection,
)

params = Params(
    dynamics={
        "seed": 42,
        "n_steps": 64,
        "particle_density": 2,
        "fluorophore_count_mode": 300,
        "fluorophore_count_var": 50**2,
        "retro_speed_mode": 2.4,
        "retro_speed_var": 0.04,
        "antero_speed_mode": 1.6,
        "antero_speed_var": 0.01,
        "velocity_noise_var": 0.01,
        "transition_matrix": {
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
        },
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
            {"name": "simplex_noise", "max_fluorophore_count_per_nm3": 0.005, "seed": 420}
        ],
    },
    imaging={
        "exposure_ms": 50,
        "truth_space": {"shape": [16, 64, 512], "scale": [0.04, 0.02, 0.02]},
        "settings": {"random_seed": 420000},
    },
)
pprint(params.model_dump(exclude={"imaging": ["channels"]}))

dynamics_rng = np.random.default_rng(seed=params.dynamics.seed)

particles = create_particle_simulators(
    **params.dynamics.model_dump(exclude=["seed", "particle_behaviour"]),
    rng=dynamics_rng
)
particle_positions, particle_fluorophore_count, particle_states = (
    run_dynamics_simulation(params.dynamics.n_steps, particles)
)

particle_path_points_um = [
    np.array(point)
    * np.array(params.imaging.truth_space.shape)
    * np.array(params.imaging.truth_space.scale)
    for point in params.rendering.particle_path_points
]
static_path = PiecewiseQuadraticBezierPath(particle_path_points_um)

digital_space_shape = (
    np.array(params.imaging.truth_space.shape) // params.imaging.output_space.downscale
).astype(int)
digital_simulation = np.zeros((params.dynamics.n_steps, *digital_space_shape))
# TODO: static distibution factory
static_distributions = [
    SimplexNoise(**params.rendering.static_distributions[0].model_dump(exclude="name"))
]
for t in range(params.dynamics.n_steps):
    time_varying_distributions = [
        ParticleSystem.on_static_path(
            static_path,
            particle_positions[t],
            particle_fluorophore_count[t],
        ),
    ]
    sim = ms.Simulation(
        **params.imaging.model_dump(),
        sample=ms.Sample(labels=time_varying_distributions + static_distributions)
    )
    digital_simulation[[t]] = sim.digital_image()


kymograph_path_points = [
    np.array(point) for point in params.kymograph.sample_path_points
]
kymo_sample_path = PiecewiseQuadraticBezierPath(kymograph_path_points)
n_path_units = int(np.floor(kymo_sample_path.length()))
n_spatial_samples = int(np.floor(kymo_sample_path.length() * 1.2))
spatial_samples = np.linspace(0, 1, n_spatial_samples)
path_samples = np.linspace(0, 1, n_path_units)
kymograph = np.zeros((params.dynamics.n_steps, n_spatial_samples))

spatial_locations = kymo_sample_path(path_samples)
sample_path_coords = np.round(spatial_locations).astype(int)

for t in range(params.dynamics.n_steps):
    time_sample = digital_simulation[
        t, sample_path_coords[:, 0], sample_path_coords[:, 1], sample_path_coords[:, 2]
    ]
    time_sample = inter_pixel_interp(
        path_samples, sample_path_coords, time_sample, new_path_samples=spatial_samples
    )
    kymograph[t] = time_sample

# --- digital sim animation
digital_display_z_index = int(digital_space_shape[0] // 2)
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
digital_animation_ax.plot(
    sample_path_coords[:, 2], sample_path_coords[:, 1], "r", alpha=0.7
)
# ---

# --- kymograph
kymo_fig, kymo_ax = plt.subplots()
kymo_ax.imshow(kymograph, cmap="gray")
kymo_ax.set_xlabel("Distance")
kymo_ax.set_ylabel("Time")
kymo_ax.set_title("Kymograph")

fig, ax = plt.subplots()
for p in range(particle_positions.shape[1]):
    ax.plot(particle_positions[:,p], np.arange(params.dynamics.n_steps))
ax.invert_yaxis()
ax.set_xlim(0, 1)
ax.set_ylim(params.dynamics.n_steps, 0)
ax.set_aspect(1 / kymograph.shape[1])


plt.show()
