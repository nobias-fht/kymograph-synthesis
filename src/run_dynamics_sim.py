if __name__ == "__main__":

    import matplotlib.pyplot as plt
    import numpy as np
    from kymograph_synthesis import dynamics

    particle_positions = dynamics.system.gen_simulation_data(
        30,
        100,
        speed_mean=1 / 100,
        speed_std=1 / 100,
        velocity_noise_std=0.002,
        state_change_prob=0.05,
        positive_prob=0.5,
        stopped_prob=0.5,
        expected_lifetime=0.5
    )
    for i in range(particle_positions.shape[1]):
        plt.plot(particle_positions[:, i], np.arange(particle_positions.shape[0]))
    plt.gca().invert_yaxis()
    plt.ylabel("Time")
    plt.xlabel("Position")

    plt.show()
