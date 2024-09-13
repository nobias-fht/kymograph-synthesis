import numpy as np
from numpy.typing import NDArray, ArrayLike

rng = np.random.default_rng()


def gen_simulation_data(
    n_particles: int,
    n_frames: int,
    speed_mean: float = 0.01,
    speed_std: float = 0.0001,
    positive_velocity_ratio=0.5,
    velocity_noise_std=0.0001,
) -> NDArray:
    initial_velocities = gen_initial_velocities(
        n_particles,
        mean=speed_mean,
        std=speed_std,
        positive_ratio=positive_velocity_ratio,
    )
    initial_positions = gen_initial_positions(n_particles)
    velocities = gen_velocities(
        initial_velocities, n_frames, noise_std=velocity_noise_std
    )
    positions = gen_positions(initial_positions, velocities)
    return positions


def gen_positions(initial_positions: NDArray, velocities: NDArray) -> NDArray:
    return initial_positions + np.cumsum(velocities, axis=0)


# TODO: offset start time instead of allowing out of line bounds ?
def gen_initial_positions(n_particles: int) -> NDArray:
    return rng.uniform(size=n_particles, low=-0.5, high=1.5)


def gen_initial_velocities(
    n_particles: int, mean: float, std: float, positive_ratio
) -> NDArray:
    speeds = rng.normal(size=n_particles, loc=mean, scale=std)
    # TODO: catch negative speeds
    directions = rng.choice(
        a=[-1, 1], size=n_particles, p=[1 - positive_ratio, positive_ratio]
    )
    return np.array(speeds) * np.array(directions)


def gen_velocities(initial_velocities: NDArray, n_frames: int, noise_std) -> NDArray:
    velocities = np.tile(initial_velocities, (n_frames, 1))
    noise = rng.normal(size=velocities.shape, loc=0, scale=noise_std)
    return velocities + noise
