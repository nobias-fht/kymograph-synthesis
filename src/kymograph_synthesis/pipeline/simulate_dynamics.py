from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from ..params import DynamicsParams
from ..dynamics import create_particle_simulators, run_dynamics_simulation


@dataclass
class DynamicsSimOutput:
    n_steps: int
    particle_positions: NDArray[np.float_]
    particle_fluorophore_count: NDArray[np.float_]
    particle_states: NDArray[np.int_]

def simulate_dynamics(
    params: DynamicsParams,
) -> tuple[NDArray[np.float_], NDArray[np.float_]]:
    dynamics_rng = np.random.default_rng(seed=params.seed)
    particles = create_particle_simulators(
        **params.model_dump(exclude=["seed", "particle_behaviour"]), rng=dynamics_rng
    )
    particle_positions, particle_fluorophore_count, particle_states = (
        run_dynamics_simulation(params.n_steps, particles)
    )
    return DynamicsSimOutput(
        n_steps=params.n_steps,
        particle_positions=particle_positions, 
        particle_fluorophore_count=particle_fluorophore_count, 
        particle_states=particle_states
    )
