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

    P_transpose = P.T
    A = P_transpose - np.eye(n)
    # to constrain sum of stationary state p to equal 1
    A = np.vstack([A, np.ones(n)])

    b = np.zeros(n)
    b = np.append(b, 1)  # same constraint as above

    # solve
    stationary_distribution = np.linalg.lstsq(A, b, rcond=None)[0]

    return {key: val for key, val in zip(keys, stationary_distribution)}


def calc_markov_transition_matrix(
    state_switch_prob: dict[MotionStateCollection, float],
    transition_prob_matrix: TransitionMatrixType,
) -> TransitionMatrixType:
    keys = list(transition_prob_matrix.keys())
    # new matrix to account for state not switching
    markov_transition_matrix = {
        key_i: { 
            key_j: (
                (1 - state_switch_prob[key_i])
                if key_i == key_j
                else transition_prob_matrix[key_i][key_j] * state_switch_prob[key_i]
            )
            for key_j in keys
        }
        for key_i in keys
    }
    return markov_transition_matrix


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


def create_particle_simulators(
    particle_density: float,
    antero_speed_mode: float,
    antero_speed_var: float,
    retro_speed_mode: float,
    retro_speed_var: float,
    velocity_noise_std: float,
    state_switch_prob: dict[MotionStateCollection, float],
    transition_prob_matrix: TransitionMatrixType,
    n_steps: int = 256,
) -> list[ParticleSimulator]:

    markov_matrix = calc_markov_transition_matrix(
        state_switch_prob, transition_prob_matrix
    )
    markov_stationary_state = calc_markov_stationary_state(markov_matrix)

    approx_travel_distance = max(antero_speed_mode, retro_speed_mode) * n_steps
    buffer_distance = approx_travel_distance * 1.5
    path_start = 0 - buffer_distance
    path_end = 1 + buffer_distance

    n_particles = int(np.ceil((path_end - path_start) * particle_density))
    print(f"Creating {n_particles}, particles")

    return [
        ParticleSimulator(
            initial_position=np.random.uniform(low=path_start, high=path_end),
            initial_state=decide_initial_state(markov_stationary_state),
            antero_speed_distr=log_normal_distr(antero_speed_mode, antero_speed_var),
            retro_speed_distr=log_normal_distr(retro_speed_mode, retro_speed_var),
            velocity_noise_distr=partial(
                np.random.normal, loc=0, scale=velocity_noise_std
            ),
            state_switch_prob=state_switch_prob,
            transition_prob_matrix=transition_prob_matrix,
        )
        for _ in range(n_particles)
    ]


def run_simulation(
    n_steps: int, particle_simulators: list[ParticleSimulator]
) -> NDArray:
    n_particles = len(particle_simulators)
    positions = np.zeros((n_steps, n_particles))
    for i in range(n_steps):
        print(f"Simulation step {i}")
        for j, particle in enumerate(particle_simulators):
            positions[i, j] = particle.position
            particle.step()

    return positions
