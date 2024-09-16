
if __name__ == "__main__":
    
    import matplotlib.pyplot as plt
    import numpy as np
    from kymograph_synthesis import dynamics
    
    particle_positions = dynamics.system.gen_simulation_data(50, 100, velocity_noise_std=0.002)
    for i in range(particle_positions.shape[1]):
        plt.plot(particle_positions[:,i], np.arange(particle_positions.shape[0]))
    plt.gca().invert_yaxis()
    plt.ylabel("Time")
    plt.xlabel("Position")

    plt.show()

    