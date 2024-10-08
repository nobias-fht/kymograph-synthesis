if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    
    from kymograph_synthesis.dynamics.particle_simulator.motion_state_collection import (
        MotionStateCollection,
    )
    from kymograph_synthesis.dynamics.system_simulator import (
        create_particle_simulators,
        run_simulation,
    )
    from kymograph_synthesis.render.ground_truth import render
    from kymograph_synthesis.render.static_path import LinearPath

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
        intensity_half_life_mode=n_steps,
        intensity_half_life_var=n_steps/4,
        velocity_noise_std=velocity_noise_std,
        state_switch_prob=switch_prob,
        transition_prob_matrix=transition_matrix,
        n_steps=n_steps,
    )

    particle_positions, particle_intensities = run_simulation(n_steps, particles)    
    static_path = LinearPath(start=np.array([16, 16]), end=np.array([16, 128 - 16]))
    space = render(
        resolution=(32, 128),
        static_path=static_path,
        positions=particle_positions,
        intensities=particle_intensities
    )

    print(space.max(), space.min())

    # Set up the figure and axis
    fig, ax = plt.subplots()
    im = ax.imshow(space[0], cmap='viridis', interpolation='nearest', vmin=space.min(), vmax=space.max())

    # Update function for the animation
    def update(frame):
        im.set_array(space[frame])
        return [im]

    # Create the animation
    anim = FuncAnimation(fig, update, frames=len(space), interval=200, blit=True, repeat=True)

    # Display the animation
    plt.show()

