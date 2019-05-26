import gym
import matplotlib
import numpy as np
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from tqdm import tqdm

matplotlib.use('Agg')

import matplotlib.pyplot as plt


def gen_fetch():
    # 
    def make_fetch_env(rank):
        def _thunk():
            env = gym.make("FetchPush-v1")
            env.seed(rank)
            env = QposWrapper(env)
            return env

        return _thunk

    start_index = 0
    num_env = 128

    env = SubprocVecEnv([make_fetch_env(i + start_index) for i in range(num_env)])

    trajs = []
    actions = []

    for i in tqdm(range(1000)):
        traj = []
        obs = env.reset()
        action = np.random.uniform(-1., 1., (num_env, 100, 4))

        for t in range(100):
            ob, _, done, _, = env.step(action[:, t])
            traj.append(ob)

        traj = np.stack(traj, axis=1)

        trajs.append(traj)
        actions.append(action)

    trajs = np.concatenate(trajs, axis=0)
    actions = np.concatenate(actions, axis=0)

    np.savez("push.npz", obs=trajs, action=actions)


class QposWrapper(gym.Wrapper):
    def reset(self):
        self.env.reset()
        return self.get_obs()

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        ob = self.get_obs()
        return ob, reward, done, info

    def get_obs(self):
        return self.env.sim.data.qpos.copy()


class ReacherWrapper(gym.Wrapper):
    def reset(self):
        self.env.reset()
        return self.get_obs()

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        ob = self.get_obs()
        return ob, reward, done, info

    def get_obs(self):
        qpos = self.env.sim.data.qpos.copy()
        qvel = self.env.sim.data.qvel.copy()
        obs = np.concatenate([qpos[:2], qvel[:2]], axis=0)
        return obs


def gen_reacher():
    # 
    def make_fetch_env(rank):
        def _thunk():
            env = gym.make("Reacher-v2")
            env.seed(rank)
            env = ReacherWrapper(env)
            return env

        return _thunk

    start_index = 0
    num_env = 128

    env = SubprocVecEnv([make_fetch_env(i + start_index) for i in range(num_env)])

    trajs = []
    actions = []
    dones = []

    for i in tqdm(range(1000)):
        traj = []
        obs = env.reset()
        action = np.random.uniform(-1., 1., (num_env, 100, 2))
        time_dones = []

        for t in range(100):
            ob, _, done, _, = env.step(action[:, t])
            traj.append(ob)
            time_dones.append(done)

        time_dones = np.array(time_dones)

        traj = np.stack(traj, axis=1)

        trajs.append(traj)
        actions.append(action)
        dones.append(time_dones)

    dones = np.concatenate(dones, axis=0)

    trajs = np.concatenate(trajs, axis=0)
    actions = np.concatenate(actions, axis=0)

    print(trajs.shape)
    print(actions.shape)
    np.savez("reacher.npz", obs=trajs, action=actions, dones=dones)


def gen_simple():
    # Free form movement on entire unit square from -1 to 1
    batch_size = 1000000
    traj_length = 100
    ob = np.random.uniform(-1, 1, (batch_size, 2))
    action = np.random.uniform(-0.05, 0.05, (batch_size, traj_length, 2))

    total_perb = np.cumsum(action, axis=1)
    total_traj = ob[:, None, :] + total_perb

    np.savez("point.npz", obs=total_traj, action=action * 20.)


def oob(x):
    return (x < -1) | (x > 1)


def is_maze_valid(dat):
    # Generate an indicator function for whether a point is valid in a hand designed
    # maze given dat (n x 2) array of data_points

    segs = 4
    oob_mask = np.any(oob(dat), axis=1)
    # dat = dat[~data_mask]

    dat_idx = ((dat[:, 0] + 1) * segs).astype(np.int32)

    data_mask = ((dat_idx % 2) == 0) | (((dat_idx % 4) == 1) & (dat[:, 1] > 0.7)) | (
                ((dat_idx % 4) == 3) & (dat[:, 1] < -0.7))

    comb_mask = (~oob_mask) & data_mask

    return comb_mask


def gen_maze():
    # Generate a dataset with of obstacles through which particles are not able to 
    # move through
    batch_size = 2000000
    traj_length = 100
    ob = np.random.uniform(-1, 1, (batch_size, 2))
    ob_mask = is_maze_valid(ob)
    ob = ob[ob_mask]

    plt.scatter(ob[:, 0], ob[:, 1])
    plt.savefig("maze.png")
    plt.clf()

    obs = [ob.copy()]
    actions = np.random.uniform(-0.05, 0.05, (ob.shape[0], traj_length, 2))

    for i in range(1, traj_length):
        new_ob = ob + actions[:, i]
        ob_mask = is_maze_valid(new_ob)

        ob[ob_mask] = new_ob[ob_mask]
        obs.append(ob.copy())

    obs = np.stack(obs, axis=1)
    np.savez("maze.npz", obs=obs, action=actions * 20.)

    plt.plot(obs[0, :, 0], obs[0, :, 1])
    plt.savefig("trajs.png")


if __name__ == "__main__":
    # gen_simple()
    # gen_maze()
    gen_reacher()
    # gen_fetch()
