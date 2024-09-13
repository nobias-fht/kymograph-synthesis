if __name__ == "__main__":
    import numpy as np
    from numpy.typing import NDArray
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from microsim import schema as ms

    from kymograph_synthesis import render


    transport = render.render_linear(
        resolution=(64, 512),
        path_start=(64-16, 64-16),
        path_end=(16, 512 - 16),
        n_particles=30,
        n_frames=120,
        speed_mean=1 / 120,
        speed_std=0.0001 / 120,
    )

    time_sim_list: list[NDArray] = []
    for i in range(transport.shape[0]):
        frame = np.zeros(shape=(4, *transport.shape[1:]))
        frame[2, ...] = transport[i]
        sim = ms.Simulation.from_ground_truth(
            frame,
            scale=(0.02, 0.01, 0.01),
            output_space={"downscale": 4},
            modality=ms.Confocal(),
            # detector=ms.CameraCCD(qe=0.82, read_noise=2, bit_depth=12, offset=100),
        )
        optical_img = sim.optical_image()
        digital_image = sim.digital_image(optical_img, with_detector_noise=True)
        print(digital_image.shape)
        time_sim_list.append(digital_image.to_numpy()[0])

    time_sim = np.concatenate(time_sim_list)

    # Set up the figure and axis
    fig, ax = plt.subplots()
    im = ax.imshow(time_sim[0], cmap='viridis', interpolation='nearest')

    # Update function for the animation
    def update(frame):
        im.set_array(time_sim[frame])
        return [im]

    # Create the animation
    anim = FuncAnimation(fig, update, frames=len(time_sim), interval=200, blit=True, repeat=True)

    # Display the animation
    plt.show()
