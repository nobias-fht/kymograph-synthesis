from pprint import pprint
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec
from microsim import schema as ms

from kymograph_synthesis.dynamics.particle_simulator.motion_state_collection import (
    MotionStateCollection,
)
from kymograph_synthesis.dynamics.system_simulator import (
    create_particle_simulators,
    run_dynamics_simulation,
)
from kymograph_synthesis.render.static_path import PiecewiseQuadraticBezierPath
from kymograph_synthesis.render.fluorophore_distributions import (
    ParticleSystem,
    SimplexNoise,
)

# TODO: sort out param imports
from kymograph_synthesis.params.params import Params
from kymograph_synthesis.params import DynamicsParams
from kymograph_synthesis.sample_kymograph import inter_pixel_interp

def convert_to_unitless_params(dynamics_params: DynamicsParams) -> dict:
    unitless_params = dynamics_params.model_dump()
    del unitless_params["path_length"]
    del unitless_params["time_delta"]
    del unitless_params["particle_behaviour"]
    del unitless_params["seed"]
    path_length = dynamics_params.path_length
    time_delta = dynamics_params.time_delta
    speed_param_names = [
        "retro_mode",
        "retro_var",
        "antero_mode",
        "antero_var",
        "noise_var",
    ]
    for speed_param in speed_param_names:
        unitless_params[speed_param] *= time_delta / path_length
    unitless_params["particle_density"] *= path_length
    return unitless_params


params = Params(dynamics={"particle_behaviour": "unidirectional"})
params.imaging.exposure_ms = 100
pprint(params.model_dump())

dynamics_rng = np.random.default_rng(seed=params.dynamics.seed)
particles = create_particle_simulators(
    **convert_to_unitless_params(params.dynamics),
    rng=dynamics_rng
)
particle_positions, particle_fluorophore_count, particle_states = run_dynamics_simulation(
    params.dynamics.n_steps, particles
)
print(particle_fluorophore_count)

path_points = [
    np.array(point) for point in params.rendering.particle_path_points
]
static_path = PiecewiseQuadraticBezierPath(path_points)

# ---
indices = static_path(np.linspace(0, 1, 128))
ax = plt.figure().add_subplot(projection='3d')
ax.plot(indices[:,0], indices[:,2], indices[:,1])
ax.set_xlim(0, None)
ax.set_ylim(0, None)
ax.set_zlim(0, None)
ax.set_aspect("equal")
# ---

truth_space_shape = params.imaging.truth_space.shape
truth_space_scale = params.imaging.truth_space.scale
downscale = params.imaging.output_space.downscale
digital_sim_shape = tuple(dim//downscale for dim in truth_space_shape)
digital_simulation = np.zeros((params.dynamics.n_steps, *digital_sim_shape))

static_distributions = [
    SimplexNoise(**params.rendering.static_distributions[0].model_dump(exclude="name"))
]
for t in range(params.dynamics.n_steps):
    time_varying_distributions = [
        ParticleSystem.on_static_path(
            static_path, particle_positions[t], particle_fluorophore_count[t]
        ),
    ]
    sim = ms.Simulation(
        **params.imaging.model_dump(),
        sample=ms.Sample(labels=static_distributions + time_varying_distributions),
    )
    digital_image = sim.digital_image()
    digital_simulation[[t]] = digital_image.data

kymo_path_points = [np.array(point) for point in params.kymograph.sample_path_points]
kymo_sample_path = PiecewiseQuadraticBezierPath(points=kymo_path_points)
path_samples = np.linspace(0, 1, params.kymograph.n_samples)
kymograph = np.zeros((params.dynamics.n_steps, params.kymograph.n_samples))
for t in range(params.dynamics.n_steps):
    sample_points = kymo_sample_path(path_samples)
    indices = np.round(sample_points).astype(int)
    time_sample = digital_simulation[t, indices[:, 0], indices[:, 1], indices[:, 2]]
    time_sample = inter_pixel_interp(path_samples, indices, time_sample, new_path_samples=path_samples)
    kymograph[t] = time_sample

fig = plt.figure()
digital_animation_ax = fig.add_subplot()

# --- digital sim animation
z_dim = truth_space_shape[0]
digital_animation_img = digital_animation_ax.imshow(
    digital_simulation[0, np.round(z_dim / 2 / downscale).astype(int)],
    cmap="gray",
    interpolation="none",
    vmin=digital_simulation.min(),
    vmax=digital_simulation.max(),
)
path_index_coords = kymo_sample_path(np.linspace(0, 1, params.kymograph.n_samples))
digital_animation_ax.plot(path_index_coords[:,2], path_index_coords[:,1], "r", linewidth=1, alpha=0.5)

def update_digital_animation(frame):
    digital_animation_img.set_array(
        digital_simulation[frame, np.round(z_dim / 2 / downscale).astype(int)]
    )
    return digital_animation_img

digital_animation = FuncAnimation(
    fig, update_digital_animation, frames=params.dynamics.n_steps, interval=200, repeat=True
)

plt.figure()
plt.imshow(kymograph, "gray")

plt.show()