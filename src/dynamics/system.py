
import numpy as np
from numpy.typing import NDArray, ArrayLike

rng = np.random.default_rng()


def gen_simulation_data(
    n_particles: int, n_frames: int, velocity_noise_std=0.0001
) -> NDArray:
    initial_velocities = gen_initial_velocities(n_particles)
    initial_positions = gen_initial_positions(n_particles)
    velocities = gen_velocities(
        initial_velocities, n_frames, noise_std=velocity_noise_std
    )
    positions = gen_positions(initial_positions, velocities)
    return positions


def gen_positions(initial_positions: NDArray, velocities: NDArray) -> NDArray:
    return initial_positions + np.cumsum(velocities, axis=0)


# TODO need to account for particles starting out of frame?
# - see what it looks like once lifetimes are added
def gen_initial_positions(n_particles: int) -> NDArray:
    return rng.uniform(size=n_particles, low=-0.5, high=1.5)


def gen_initial_velocities(
    n_particles: int, mean: float = 0.01, std: float = 0.0001, positive_ratio=0.5
) -> NDArray:
    velocities = rng.normal(size=n_particles, loc=mean, scale=std)
    random_signs = rng.choice(
        a=[-1, 1], size=n_particles, p=[1 - positive_ratio, positive_ratio]
    )
    return np.array(velocities) * np.array(random_signs)


def gen_velocities(
    initial_velocities: NDArray, n_frames: int, noise_std=0.0001
) -> NDArray:
    velocities = np.tile(initial_velocities, (n_frames, 1))
    noise = rng.normal(size=velocities.shape, loc=0, scale=noise_std)
    return velocities + noise
