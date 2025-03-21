from typing import TypedDict, Mapping
from typing_extensions import Unpack

import numpy as np
from numpy.typing import NDArray, ArrayLike
from ...params import DynamicsParams
from ...dynamics import create_particle_simulators, run_dynamics_simulation


class DynamicsSimOutput(TypedDict):
    n_steps: int
    particle_positions: NDArray[np.float_]
    particle_fluorophore_count: NDArray[np.float_]
    particle_states: NDArray[np.int_]


def simulate_dynamics(
    params: DynamicsParams,
) -> DynamicsSimOutput:
    dynamics_rng = np.random.default_rng(seed=params.seed)
    particles = create_particle_simulators(
        **params.model_dump(exclude={"seed", "particle_behaviour"}), rng=dynamics_rng
    )
    particle_positions, particle_fluorophore_count, particle_states = (
        run_dynamics_simulation(params.n_steps, particles)
    )

    return {
        "n_steps": params.n_steps,
        "particle_positions": particle_positions,
        "particle_fluorophore_count": particle_fluorophore_count,
        "particle_states": particle_states,
    }
