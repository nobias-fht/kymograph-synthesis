from typing import TypedDict
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray
from ...params import DynamicsParams
from ...dynamics import create_particle_simulators, run_dynamics_simulation
from ...dynamics.particle_simulator.particle_simulator import ParticleSimulator


class DynamicsSimOutput(TypedDict):
    n_steps: int
    particle_positions: NDArray[np.float_]
    particle_fluorophore_count: NDArray[np.float_]
    particle_states: NDArray[np.int_]


def simulate_dynamics(
    params: DynamicsParams | Sequence[DynamicsParams], n_steps: int
) -> DynamicsSimOutput:
    if isinstance(params, DynamicsParams):
        params = [params]

    particles: list[ParticleSimulator]
    all_positions: list[NDArray] = []
    all_fluorophore_count: list[NDArray] = []
    all_states: list[NDArray] = []
    for param_set in params:
        dynamics_rng = np.random.default_rng(seed=param_set.seed)
        particles = create_particle_simulators(
            n_steps=n_steps,
            **param_set.model_dump(exclude={"seed", "particle_behaviour"}),
            rng=dynamics_rng
        )
        particle_positions, particle_fluorophore_counts, particle_states = (
            run_dynamics_simulation(n_steps, particles)
        )
        all_positions.append(particle_positions)
        all_fluorophore_count.append(particle_fluorophore_counts)
        all_states.append(particle_states)

    return DynamicsSimOutput(
        n_steps=n_steps,
        particle_positions=np.concatenate(all_positions, axis=1),
        particle_fluorophore_count=np.concatenate(all_fluorophore_count, axis=1),
        particle_states=np.concatenate(all_states, axis=1),
    )
