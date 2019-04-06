import numpy as np


if __name__ == "__main__":
    # Generate a dataset of random particle movement by perturbation of an action

    batch_size = 50000
    traj_length = 100
    ob = np.random.uniform(-1, 1, (batch_size, 2))
    action = np.random.uniform(-0.05, 0.05, (batch_size, traj_length, 2))

    total_perb = np.cumsum(action, axis=1)
    total_traj = ob[:, None, :] + total_perb

    np.savez("point.npz", obs=total_traj, action=action)
