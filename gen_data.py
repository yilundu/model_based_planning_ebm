import numpy as np

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt


def gen_simple():
    # Free form movement on entire unit square from -1 to 1
    batch_size = 100000
    traj_length = 100
    ob = np.random.uniform(-1, 1, (batch_size, 2))
    action = np.random.uniform(-0.05, 0.05, (batch_size, traj_length, 2))

    total_perb = np.cumsum(action, axis=1)
    total_traj = ob[:, None, :] + total_perb

    np.savez("point.npz", obs=total_traj, action=action)

def oob(x):
    return (x < -1) | (x > 1)

def is_maze_valid(dat):
    # Generate an indicator function for whether a point is valid in a hand designed
    # maze given dat (n x 2) array of data_points

    segs = 10
    oob_mask = np.any(oob(dat), axis=1)
    # dat = dat[~data_mask]

    dat_idx = ((dat[:, 0] + 1) * segs).astype(np.int32)

    data_mask = ((dat_idx % 2) == 0) | (((dat_idx % 4) == 1) & (dat[:, 1] > 0.7)) | (((dat_idx % 4) == 3) & (dat[:, 1] < -0.7))

    comb_mask = (~oob_mask) & data_mask

    return comb_mask

def gen_maze():
    # Generate a dataset with of obstacles through which particles are not able to 
    # move through
    batch_size = 200000
    traj_length = 100
    ob = np.random.uniform(-1, 1, (batch_size, 2))
    ob_mask = is_maze_valid(ob)
    ob = ob[ob_mask]

    obs = [ob.copy()]
    actions = np.random.uniform(-0.1, 0.1, (ob.shape[0], traj_length, 2))

    for i in range(1, traj_length):
        new_ob = ob + actions[:, i-1]
        ob_mask = is_maze_valid(new_ob)

        ob[ob_mask] = new_ob[ob_mask]
        obs.append(ob.copy())

    obs = np.stack(obs, axis=1)
    np.savez("maze.npz", obs=obs, action=actions)

    plt.plot(obs[0, :, 0], obs[0, :, 1])
    plt.savefig("trajs.png")


if __name__ == "__main__":
    gen_maze()
