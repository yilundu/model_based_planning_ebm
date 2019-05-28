import argparse
from multiprocessing.pool import Pool

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('-save_path', type=str, default='./data/')
parser.add_argument('--task_type', choices=['hand', 'fetch'])

args = parser.parse_args()


def gen_instance(arg):
    from gpr.envs.dactyl_reach import make_env
    from gpr.envs import dactyl_locked
    from gpr.storage.s3_policy_saver import load_policy
    i, task = arg

    if task == 'fetch':
        policy = load_policy('rapid_policies/ci/ci-run-ddpg-rubik-reach-1bd9565d/evaluator/policy_latest.npz')
        env = make_env(randomize=True, her_support=True)
    elif task == 'hand':
        policy = load_policy('rapid_policies/ci/ci-run-ddpg-rubik-xyz-1bd9565d/evaluator/policy_latest.npz')
        env = dactyl_locked.make_env(randomize=True, her_support=True)

    obs = env.reset()
    for _ in range(i):
        obs = env.reset()

    # env.reset(force_seed=i)

    np.random.seed(i)
    sim = env.unwrapped.sim

    obs_list = []
    action_list = []

    for i in range(500):
        traj_action = []
        traj_obs = []

        for j in range(100):
            # random sampling of actions
            # action = env.action_space.sample()

            # action ~ pi

            # if (j % 4) == 0:
            action_random = env.action_space.sample()
            # else:

            action, _ = policy.act(obs)
            env.step(action + 0.1 * action_random)

            obs, _, _, _ = env.step(action)

            traj_action.append(np.copy(action))
            traj_obs.append(np.copy(env.unwrapped.sim.data.qpos))

            data_list = dir(env.unwrapped.sim.data)

            # For messing
            # print(sim.data.qpos)
            # sim.data.qpos[:] = np.random.random(size=sim.data.qpos.shape)
            # sim.forward()

        traj_obs = np.array(traj_obs)
        traj_action = np.array(traj_action)

        obs_list.append(traj_obs)
        action_list.append(traj_action)

        obs = env.reset()
        policy.reset()

    obs = np.array(obs_list)
    action = np.array(action_list)

    return (obs, action)


def gen(task):
    n = 400
    args = zip(list(range(n)), [task] * n)
    pool = Pool()
    dat = pool.map(gen_instance, args)
    obs, action = zip(*dat)
    obs = np.concatenate(obs, axis=0)
    action = np.concatenate(action, axis=0)

    np.savez(args.save_path + "{}.npz".format(task), obs=obs, action=action)


if __name__ == "__main__":
    # Task options are fetch and hand
    gen(args.task_type)
