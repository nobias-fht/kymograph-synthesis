if __name__ == "__main__":
    import numpy as np
    from numpy.typing import NDArray
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from microsim import schema as ms

    from kymograph_synthesis import render
    from kymograph_synthesis.render.static_path import LinearPath

    n_frames = 120
    n_particles = 50
    path_start = (64-16, 64-16)
    path_end = (16, 512 - 16)

    transport = render.render_linear(
        resolution=(64, 512),
        path_start=path_start,
        path_end=path_end,
        n_particles=n_particles,
        n_frames=n_frames,
        speed_mean=2 / n_frames,
        speed_std=0.1 / n_frames,
        # speed_std=0.5 / n_frames,
        velocity_noise_std=0.002,
        state_change_prob=0,
        # state_change_prob=0.05,
        positive_prob=0.5,
        # stopped_prob=0.5,
        stopped_prob=0,
        expected_lifetime=1
    )
    transport = transport * 100

    time_sim_list: list[NDArray] = []
    for i in range(transport.shape[0]):
        frame = np.zeros(shape=(4, *transport.shape[1:]))
        frame[2, ...] = transport[i]
        sim = ms.Simulation.from_ground_truth(
            frame,
            scale=(0.4, 0.02, 0.02),
            output_space={"downscale": 4},
            # modality=ms.Confocal(),
            modality=ms.Widefield(),
            detector=ms.CameraCCD(qe=1, read_noise=4, bit_depth=12, offset=100),
        )
        optical_img = sim.optical_image()

        digital_image = sim.digital_image(optical_img, exposure_ms=100 ,with_detector_noise=True)
        print(digital_image.shape)
        time_sim_list.append(digital_image.to_numpy()[0])
    time_sim = np.concatenate(time_sim_list)

    n_spatial_samples = 128
    spatial_samples = np.linspace(0, 1, n_spatial_samples)
    kymograph = np.zeros((n_frames, n_spatial_samples))
    linear_path = LinearPath(start=np.array(path_start)/4, end=np.array(path_end)/4)
    for i in range(n_frames):
        spatial_locations = linear_path(spatial_samples)
        coords = np.round(spatial_locations).astype(int)
        print(coords.shape)
        time_sample = time_sim[i, coords[0], coords[1]]
        kymograph[i] = time_sample


    plt.figure()
    plt.imshow(kymograph, "gray")
    plt.imsave("/Users/melisande.croft/Desktop/kymograph-uni2.png", kymograph, cmap="gray")

    # Set up the figure and axis
    fig, ax = plt.subplots()
    im = ax.imshow(time_sim[0], cmap='viridis', interpolation='nearest', vmin=time_sim.min(), vmax=time_sim.max())

    # Update function for the animation
    def update(frame):
        im.set_array(time_sim[frame])
        return [im]

    # Create the animation
    anim = FuncAnimation(fig, update, frames=len(time_sim), interval=200, blit=True, repeat=True)

    # Display the animation
    plt.show()
