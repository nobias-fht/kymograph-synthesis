from kymograph_synthesis.dynamics.particle_simulator.motion_state_collection import (
    MotionStateCollection,
)
from kymograph_synthesis.dynamics.system_simulator import (
    create_particle_simulators,
    run_simulation,
)

if __name__ == "__main__":

    import numpy as np
    import matplotlib.pyplot as plt

    n_steps = 120
    particle_density = 30

    retro_speed_mode = 0.4e-2
    retro_speed_var = 0
    antero_speed_mode = 1.6e-2
    antero_speed_var = 0
    velocity_noise_mean = 0
    velocity_noise_std = 0.2e-2

    switch_prob = {
        MotionStateCollection.ANTEROGRADE: 0.01,
        MotionStateCollection.STATIONARY: 0.1,
        MotionStateCollection.RETROGRADE: 0.1,
    }
    transition_matrix = {
        MotionStateCollection.ANTEROGRADE: {
            MotionStateCollection.ANTEROGRADE: 0,
            MotionStateCollection.STATIONARY: 0.5,
            MotionStateCollection.RETROGRADE: 0.5,
        },
        MotionStateCollection.STATIONARY: {
            MotionStateCollection.ANTEROGRADE: 0.1,
            MotionStateCollection.STATIONARY: 0,
            MotionStateCollection.RETROGRADE: 0.9,
        },
        MotionStateCollection.RETROGRADE: {
            MotionStateCollection.ANTEROGRADE: 0.1,
            MotionStateCollection.STATIONARY: 0.9,
            MotionStateCollection.RETROGRADE: 0,
        },
    }

    particles = create_particle_simulators(
        particle_density,
        antero_speed_mode,
        antero_speed_var,
        retro_speed_mode,
        retro_speed_var,
        intensity_mode=100,
        intensity_var=10,
        intensity_half_life_mode=n_steps*4,
        intensity_half_life_var=n_steps,
        velocity_noise_std=velocity_noise_std,
        state_switch_prob=switch_prob,
        transition_prob_matrix=transition_matrix,
        n_steps=n_steps,
    )

    particle_positions, particle_intensities = run_simulation(n_steps, particles)

    for i in range(particle_positions.shape[1]):
        plt.plot(particle_positions[:, i], np.arange(particle_positions.shape[0]))
    plt.gca().invert_yaxis()
    plt.ylabel("Time")
    plt.xlabel("Position")
    plt.xlim(0, 1)

    plt.figure()
    for i in range(particle_intensities.shape[1]):
        plt.plot(np.arange(n_steps), particle_intensities[:, i])
    plt.xlabel("Time")
    plt.ylabel("Intensity")

    plt.show()
