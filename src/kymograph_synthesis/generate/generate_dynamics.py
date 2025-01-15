import numpy as np
from ..params import DynamicsParams
from ..dynamics import create_particle_simulators, run_dynamics_simulation


def generate_dynamics(params: DynamicsParams):
    dynamics_rng = np.random.default_rng(seed=params.seed)
    particles = create_particle_simulators(
        **params.model_dump(exclude=["seed", "particle_behaviour"]), rng=dynamics_rng
    )
    particle_positions, particle_fluorophore_count, particle_states = (
        run_dynamics_simulation(params.n_steps, particles)
    )
    return particle_positions, particle_fluorophore_count, particle_states 
