if __name__ == "__main__":
    import numpy as np
    from numpy.typing import NDArray
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from microsim import schema as ms
    import opensimplex

    from kymograph_synthesis import render
    from kymograph_synthesis.render.static_path import LinearPath

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
    from kymograph_synthesis.render.space_time_objects import (
        SpaceTimeObject,
        ParticleSystem,
        StaticSimplexNoise,
    )

    n_steps = 120
    particle_density = 30

    retro_speed_mode = 0.4e-2
    retro_speed_var = 0
    antero_speed_mode = 1.6e-2
    antero_speed_var = 0
    velocity_noise_mean = 0
    velocity_noise_std = 0.2e-2

    intensity_mode = 100
    intensity_var = 10

    path_start = np.array([2, 64 - 16, 64 - 16])
    path_end = np.array([2, 16, 512 - 16])

    n_spatial_samples = 128

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
        intensity_mode=intensity_mode,
        intensity_var=intensity_var,
        intensity_half_life_mode=n_steps,
        intensity_half_life_var=n_steps / 4,
        velocity_noise_std=velocity_noise_std,
        state_switch_prob=switch_prob,
        transition_prob_matrix=transition_matrix,
        n_steps=n_steps,
    )

    particle_positions, particle_intensities = run_simulation(n_steps, particles)
    static_path = LinearPath(start=path_start, end=path_end)

    objects: list[SpaceTimeObject] = [
        ParticleSystem.on_static_path(
            static_path, particle_positions, particle_intensities
        ),
        StaticSimplexNoise(
            scales=[5, 10], scale_weights=[1, 1], max_intensity=intensity_mode / 256
        ),
    ]
    z_dim = 4
    space_time_gt = np.zeros((n_steps, z_dim, 64, 512))
    for t in range(n_steps):
        for object in objects:
            object.render(t, space_time_gt[t])

    time_sim_list: list[NDArray] = []
    for i, frame in enumerate(space_time_gt):
        sim = ms.Simulation.from_ground_truth(
            frame,
            scale=(0.4, 0.02, 0.02),
            output_space={"downscale": 4},
            # modality=ms.Confocal(),
            modality=ms.Widefield(),
            detector=ms.CameraCCD(qe=1, read_noise=4, bit_depth=12, offset=100),
        )
        optical_img = sim.optical_image()

        digital_image = sim.digital_image(
            optical_img, exposure_ms=100, with_detector_noise=True
        )
        print(digital_image.shape)
        time_sim_list.append(digital_image.to_numpy()[0])
    time_sim = np.concatenate(time_sim_list)

    spatial_samples = np.linspace(0, 1, n_spatial_samples)
    kymograph = np.zeros((n_steps, n_spatial_samples))
    linear_path = LinearPath(start=np.array(path_start) / 4, end=np.array(path_end) / 4)
    for t in range(n_steps):
        spatial_locations = linear_path(spatial_samples)
        coords = np.round(spatial_locations).astype(int)
        print(coords.shape)
        time_sample = time_sim[t, coords[:, 1], coords[:, 2]]
        kymograph[t] = time_sample

    print("background", objects[1].noise_array.min(), objects[1].noise_array.max())

    plt.figure()
    for i in range(particle_positions.shape[1]):
        plt.plot(particle_positions[:, i], np.arange(particle_positions.shape[0]))
    plt.gca().invert_yaxis()
    plt.ylabel("Time")
    plt.xlabel("Position")
    plt.xlim(0, 1)

    plt.figure()
    plt.imshow(kymograph, "gray")
    plt.imsave(
        "/Users/melisande.croft/Desktop/kymograph-uni2.png", kymograph, cmap="gray"
    )

    # Set up the figure and axis
    fig, ax = plt.subplots()
    im = ax.imshow(
        time_sim[0],
        cmap="viridis",
        interpolation="nearest",
        vmin=time_sim.min(),
        vmax=time_sim.max(),
    )

    # Update function for the animation
    def update(frame):
        im.set_array(time_sim[frame])
        return [im]

    # Create the animation
    anim = FuncAnimation(
        fig, update, frames=len(time_sim), interval=200, blit=True, repeat=True
    )

    # Display the animation
    plt.show()
