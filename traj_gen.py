import numpy as np
from tqdm import tqdm

from multiprocessing.pool import Pool


def fetch_gen_instance(i):
    from gpr.envs.dactyl_reach import make_env
    from gpr.storage.s3_policy_saver import load_policy

    policy = load_policy('rapid_policies/ci/ci-run-ddpg-rubik-reach-1bd9565d/evaluator/policy_latest.npz')
    env = make_env(randomize=False, her_support=True)
    obs = env.reset(force_seed=i)
    sim = env.unwrapped.sim

    obs_list = []
    action_list = []

    for i in range(25):
        traj_action = []
        traj_obs = []
        for j in range(100):
            # random sampling of actions
            #action = env.action_space.sample()

            # action ~ pi
            action, _ = policy.act(obs)
            obs, _, _, _ = env.step(action)

            traj_action.append(np.copy(action))
            traj_obs.append(np.copy(env.unwrapped.sim.data.qpos))

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

def fetch_gen():
    args = list(range(100))
    pool = Pool()
    dat = pool.map(fetch_gen_instance, args)
    obs, action = zip(*dat)
    obs = np.concatenate(obs, axis=0)
    action = np.concatenate(action, axis=0)

    np.savez("fetch.npz", obs=obs, action=action)


if __name__ == "__main__":
    fetch_gen()
