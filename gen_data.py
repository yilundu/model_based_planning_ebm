import argparse

import gym
import matplotlib
import numpy as np
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from tqdm import tqdm
from envs import Ball
from utils import is_maze_valid

matplotlib.use('Agg')

import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('-save_path', type=str, default='./data/')
parser.add_argument('--data_type', choices=['simple', 'maze', 'reacher', 'fetch', 'phy', 'continual_reacher'])
parser.add_argument('--phy_type', choices=['a', 'cor'], default='a')

args = parser.parse_args()


def gen_fetch():
    # generate data from FetchPush-v1 environment
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

    np.savez(args.save_path + "push.npz", obs=trajs, action=actions)


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
    # generate data from Reacher-v2 environment
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
    np.savez(args.save_path + "reacher.npz", obs=trajs, action=actions, dones=dones)


def gen_continual_reacher():
    # generate data from Reacher-v2 environment
    def make_fetch_env(rank):
        def _thunk():
            env = ContinualReacher7DOFEnv()
            env.seed(rank)
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
        action = np.random.uniform(-1., 1., (num_env, 100, 7))
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
    np.savez(args.save_path + "reacher_continual.npz", obs=trajs, action=actions, dones=dones)


def gen_simple():
    # Free form movement on entire unit square from -1 to 1
    batch_size = 1000000
    traj_length = 100
    ob = np.random.uniform(-1, 1, (batch_size, 2))
    action = np.random.uniform(-0.05, 0.05, (batch_size, traj_length, 2))

    total_perb = np.cumsum(action, axis=1)
    total_traj = ob[:, None, :] + total_perb

    np.savez(args.save_path + "point.npz", obs=total_traj, action=action * 20.)


<<<<<<< HEAD
=======
def gen_phy(phy_type):
    # generate data for ball trajectories, with physical parameters
    batch_size = 10000
    traj_length = 100

    trajs = []
    actions = []
    dones = []

    for i in tqdm(range(batch_size)):
        # create a new environment
        if phy_type == "a":
            # vary environmental force
            phy_param = np.random.uniform(-0.05, 0.05, size=2)
            env = Ball(a=phy_param, random_starts=True, phy_type=phy_type)
        elif phy_type == "cor":
            # vary coefficient of restitution
            phy_param = np.random.uniform(0.0, 1.0)
            env = Ball(cor=phy_param, random_starts=True, phy_type=phy_type)

        traj = []
        action = np.random.uniform(-1., 1., (traj_length, 2))
        time_dones = []

        ob = env.reset()

        for t in range(traj_length):
            ob, _, done, _ = env.step(action[t])
            traj.append(ob)
            time_dones.append(done)

        traj = np.array(traj)
        time_dones = np.array(time_dones)

        trajs.append(traj)
        actions.append(action)
        dones.append(time_dones)

        del env

    dones = np.array(dones)
    trajs = np.array(trajs)
    actions = np.array(actions)

    print("dones shape", dones.shape)
    print("trajs shape", trajs.shape)
    print("actions shape", actions.shape)

    np.savez(args.save_path + "phy_{}.npz".format(phy_type), obs=trajs, action=actions, dones=dones)


def oob(x):
    return (x < -1) | (x > 1)
>>>>>>> d83ad48da98cc8782a0d7432294abaaed2ec5fdf


def gen_maze():
    # Generate a dataset with of obstacles through which particles are not able to 
    # move through
    batch_size = 2000000
    traj_length = 100
    ob = np.random.uniform(-1, 1, (batch_size, 2))
    ob_mask = is_maze_valid(ob)
    ob = ob[ob_mask]

    plt.scatter(ob[:, 0], ob[:, 1])
    plt.savefig(args.save_path + "maze.png")
    plt.clf()

    obs = [ob.copy()]
    actions = np.random.uniform(-0.05, 0.05, (ob.shape[0], traj_length, 2))

    for i in range(1, traj_length):
        new_ob = ob + actions[:, i]
        ob_mask = is_maze_valid(new_ob)

        ob[ob_mask] = new_ob[ob_mask]
        obs.append(ob.copy())

    obs = np.stack(obs, axis=1)
    np.savez(args.save_path + "maze.npz", obs=obs, action=actions * 20.)

    plt.plot(obs[0, :, 0], obs[0, :, 1])
    plt.savefig(args.save_path + "trajs.png")


if __name__ == "__main__":
    if args.data_type == 'simple':
        gen_simple()
    elif args.data_type == 'maze':
        gen_maze()
    elif args.data_type == 'reacher':
        gen_reacher()
    elif args.data_type == 'fetch':
        gen_fetch()
    elif args.data_type == 'continual_reacher':
        gen_continual_reacher()
    elif args.data_type == 'phy':
        gen_phy(args.phy_type)
    else:
        raise AssertionError('Invalid data type')
