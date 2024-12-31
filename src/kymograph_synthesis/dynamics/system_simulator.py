from typing import Callable
from functools import partial

import numpy as np
from numpy.typing import NDArray
from scipy import optimize

from .particle_simulator.particle_simulator import (
    TransitionMatrixType,
    ParticleSimulator,
)
from .particle_simulator.motion_state_collection import MotionStateCollection


def calc_markov_stationary_state(
    markov_transition_matrix: TransitionMatrixType,
) -> dict[MotionStateCollection, float]:
    keys = list(markov_transition_matrix.keys())

    # solving (P.T - I)p = 0 where p is the stationary state vector

    # transition matrix as numpy
    P = np.array(
        [[markov_transition_matrix[key_i][key_j] for key_j in keys] for key_i in keys]
    )
    n = P.shape[0]  # n states

    eigenvalues, eigenvectors = np.linalg.eig(P.T) # left eigen values
    stationary_states = []
    for i, eigenvalue in enumerate(eigenvalues):
        if np.isclose(eigenvalue, 1, atol=1e-2):
            # normalize the eigenvector corresponding to eigenvalue 1
            stationary_vector = np.real(eigenvectors[:, i])
            stationary_vector /= stationary_vector.sum()
            stationary_states.append(stationary_vector)
    if len(stationary_states) == 0:
        raise ValueError("Stationary state not found for Markov chain.")

    # create random linear combination of all the stationary states
    stationary_distribution = np.zeros(n)
    state_weights = np.random.random(len(stationary_states))
    for state, weight in zip(stationary_states, state_weights):
        stationary_distribution += weight * state

    stationary_distribution /= stationary_distribution.sum()

    return {key: val for key, val in zip(keys, stationary_distribution)}



def log_normal_params(mode: float, var: float):

    eqn = lambda x, var, mode: x**4 - x**3 - (var / mode**2)
    result = optimize.root_scalar(
        f=eqn, args=(var, mode), method="toms748", bracket=[1e-16, 20]
    )
    if not result.converged:
        raise ValueError("No convergence when solving log normal params")

    sigma_2 = np.log(result.root)
    mu = np.log(mode) + sigma_2

    return mu, sigma_2**0.5


def log_normal_distr(mode: float, var: float) -> Callable[[], float]:
    mu, sigma = log_normal_params(mode, var)
    return partial(np.random.lognormal, mean=mu, sigma=sigma)


def decide_initial_state(initial_state_ratios: dict[MotionStateCollection, float]):
    decision_prob = np.random.random()
    cumulative_prob = 0
    for state, prob in initial_state_ratios.items():
        cumulative_prob += prob
        if decision_prob <= cumulative_prob:
            return state
    print("should be unreachable")


def create_particle_simulators(
    particle_density: float,
    antero_speed_mode: float,
    antero_speed_var: float,
    retro_speed_mode: float,
    retro_speed_var: float,
    intensity_mode: float,
    intensity_var: float,
    intensity_half_life_mode: float,
    intensity_half_life_var: float,
    velocity_noise_std: float,
    transition_matrix: TransitionMatrixType,
    n_steps: int = 256,
) -> list[ParticleSimulator]:

    markov_stationary_state = calc_markov_stationary_state(transition_matrix)

    approx_travel_distance = max(antero_speed_mode, retro_speed_mode) * n_steps
    buffer_distance = approx_travel_distance * 1.5
    path_start = 0 - buffer_distance
    path_end = 1 + buffer_distance

    initial_intensity_distr = log_normal_distr(intensity_mode, intensity_var)
    intensity_half_life_distr = log_normal_distr(
        intensity_half_life_mode, intensity_half_life_var
    )

    n_particles = int(np.round((path_end - path_start) * particle_density))

    return [
        ParticleSimulator(
            initial_position=np.random.uniform(low=path_start, high=path_end),
            initial_state=decide_initial_state(markov_stationary_state),
            antero_speed_distr=log_normal_distr(antero_speed_mode, antero_speed_var),
            retro_speed_distr=log_normal_distr(retro_speed_mode, retro_speed_var),
            initial_intensity=initial_intensity_distr(),
            intensity_half_life=intensity_half_life_distr(),
            velocity_noise_distr=partial(
                np.random.normal, loc=0, scale=velocity_noise_std
            ),
            transition_matrix=transition_matrix,
        )
        for _ in range(n_particles)
    ]


def run_dynamics_simulation(
    n_steps: int, particle_simulators: list[ParticleSimulator]
) -> NDArray:
    n_particles = len(particle_simulators)
    positions = np.zeros((n_steps, n_particles))
    intensities = np.zeros((n_steps, n_particles))
    states = np.zeros((n_steps, n_particles), dtype=int)
    for i in range(n_steps):
        for j, particle in enumerate(particle_simulators):
            positions[i, j] = particle.position
            intensities[i, j] = particle.intensity
            states[i, j] = particle.state.value
            particle.step()

    return positions, intensities, states
