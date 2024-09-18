if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    
    from kymograph_synthesis import render


    space = render.render_linear(
        resolution=(32, 128),
        path_start=(16, 16),
        path_end=(16, 128 - 16),
        n_particles=50,
        n_frames=120,
        speed_mean=1 / 120,
        speed_std=0.0001 / 120,
    
    )

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

