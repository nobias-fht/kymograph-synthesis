import numpy as np
from numpy.typing import NDArray, ArrayLike

rng = np.random.default_rng()


def gen_simulation_data(
    n_particles: int,
    n_frames: int,
    speed_mean: float = 0.01,
    speed_std: float = 0.0001,
    positive_velocity_probability=0.5,
    velocity_noise_std=0.0001,
) -> NDArray:
    """
    Generate the position of each particle at each frame given simulation parameters.

    Parameters
    ----------
    n_particles : int
        The number of particles to simulate.
    n_frames : int
        The number of frames to generate the simulation for.
    speed_mean : float
        The mean of the normal distribution the speeds will be sampled from. The units
        are expressed as `ratio of path / frame.` e.g. if a particle has a speed of
        0.01 in 100 frames it will travel the entire path.
    speed_std : float
        The standard deviattion of the normal distribution the speeds will be sampled
        from. The units are expressed as `ratio of path / frame.` e.g. if a particle
        has a speed of 0.01 in 100 frames it should travel the entire path.
    positive_velocity_probability : float
        The probablility that a particle will have a positive velocity.
    velocity_noise_std : float
        The standard deviation of the noise to add to the velocities, to add variance
        across frames.

    Returns
    -------
    numpy.ndarray
        2D array of floats containing the position of each particle at each frame, the
        first axis represents each frame and the second axis represents each particle.
    """
    initial_velocities = gen_initial_velocities(
        n_particles,
        mean=speed_mean,
        std=speed_std,
        positive_probability=positive_velocity_probability,
    )
    initial_positions = gen_initial_positions(n_particles)
    velocities = gen_velocities(
        initial_velocities, n_frames, noise_std=velocity_noise_std
    )
    positions = calc_positions(initial_positions, velocities)
    existence_mask = gen_existence_mask(n_particles, n_frames, n_frames/2)

    # TODO: refactor start offset 
    #   (To allow particles to start at any position at any time)
    output = np.full((n_frames, n_particles), fill_value=np.nan)
    for i in range(n_particles):
        particle_existance_mask = existence_mask[:, i]
        output[particle_existance_mask, i] = positions[:np.count_nonzero(particle_existance_mask), i]

    return output


def calc_positions(initial_positions: NDArray, velocities: NDArray) -> NDArray:
    """
    Calculate the position of each particle along the path.

    Calculated given it's initial position and it's velocity at each frame.

    Paramters
    ---------
    initial_position : numpy.ndarray
        1D array containing the intitial position of each particle.
    velocities : numpy.ndarray
        2D array of floats containing the velocity of each particle at each frame, the
        first axis represents each frame and the second axis represents each particle.

    Returns
    -------
    numpy.ndarray
        2D array of floats containing the position of each particle at each frame, the
        first axis represents each frame and the second axis represents each particle.
    """
    return initial_positions + np.cumsum(velocities, axis=0)


def gen_initial_positions(n_particles: int) -> NDArray:
    """
    Generate the initial positions of particles along a path.

    The position is expressed as a ratio, e.g. if the position is 0.5 the particle will
    be at the center of the path.

    Parameters
    ----------
    n_particles : int
        The number of particles to simulate.

    Returns
    -------
    numpy.ndarray
        1D array containing the intitial position of each particle.
    """
    return rng.uniform(size=n_particles, low=0, high=1)


def gen_initial_velocities(
    n_particles: int, mean: float, std: float, positive_probability
) -> NDArray[np.float_]:
    """
    Generate the initial velocities for a given number of particles.

    Speeds are sampled from a normal distribution, the velocities have a given
    probability to be positive or negative.

    Parameters
    ----------
    n_particles : int
        The number of particles
    mean : float
        The mean of the normal distribution the speeds will be sampled from. The units
        are expressed as `ratio of path / frame.` e.g. if a particle has a speed of
        0.01 in 100 frames it will travel the entire path.
    std : float
        The standard deviattion of the normal distribution the speeds will be sampled
        from. The units are expressed as `ratio of path / frame.` e.g. if a particle
        has a speed of 0.01 in 100 frames it should travel the entire path.
    positive_probability : float
        The probablility that a particle will have a positive velocity.

    Returns
    -------
    numpy.ndarray
        1D float array with length `n_particles`. The initial velocity for each
        particle.
    """
    # TODO: guard to ensure probability in [0, 1]
    speeds = rng.normal(size=n_particles, loc=mean, scale=std)
    # TODO: catch negative speeds
    directions = rng.choice(
        a=[-1, 1], size=n_particles, p=[1 - positive_probability, positive_probability]
    )
    return np.array(speeds) * np.array(directions)


def gen_velocities(
    initial_velocities: NDArray[np.float_], n_frames: int, noise_std: float
) -> NDArray:
    """
    Generate the velocity of each particle for `n_frames`.

    Parameters
    ----------
    initial_velocities : numpy.ndarray
        1D float array representing the initial velocity for each particle.
    n_frames : int
        The number of frames to generate the velocity for.
    noise_std : float
        The standard deviation of the noise to add to the velocities, to add variance
        across frames.

    Returns
    -------
    numpy.ndarray
        2D array of floats containing the velocity of each particle at each frame, the
        first axis represents each frame and the second axis represents each particle.
    """
    # same velocity for each frame
    velocities = np.tile(initial_velocities, (n_frames, 1))
    noise = rng.normal(size=velocities.shape, loc=0, scale=noise_std)
    return velocities + noise  # velocities with noise


def decide_lifetimes(n_particles: int, expected_lifetime: float) -> NDArray[np.float_]:
    """
    Decide the lifetime of each particle, sampled from an exponential distribution.

    Parameters
    ----------
    n_particles : int
        The number of particles
    expected_lifetime : float
        The expected lifetime, in the unit of frames.
    """
    return rng.exponential(size=n_particles, scale=expected_lifetime)


def decide_start_time(
    n_particles: int, n_frames: int, expected_lifetime: float
) -> NDArray[np.float_]:
    return rng.uniform(
        size=n_particles, low=0 - expected_lifetime, high=n_frames + expected_lifetime
    )


def gen_existence_mask(
    n_particles: int, n_frames: int, expected_lifetime: float
) -> NDArray[np.bool_]:
    mask = np.zeros((n_frames, n_particles), dtype=bool)

    start_times = decide_start_time(n_particles, n_frames, expected_lifetime).reshape(1, -1)
    lifetimes = decide_lifetimes(n_particles, expected_lifetime).reshape(1, -1)

    frame_indices, _ = np.mgrid[:n_frames, :n_particles]
    mask[(start_times < frame_indices) & (frame_indices < (start_times + lifetimes))] = True

    return mask

