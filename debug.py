import numpy as np
from trajopt.envs.continual_reacher_env import ContinualReacher7DOFEnv
import imageio


if __name__ == "__main__":
    env = ContinualReacher7DOFEnv()
    dat = np.load("data/continual_reacher.npz")
    obs = dat['obs']

    sim = env.sim
    ims = []
    for i in range(obs.shape[1]):
        sim.data.qpos[:7] = obs[3, i, :7]
        sim.forward()
        im = sim.render(256, 256)
        ims.append(im)

    imageio.mimwrite("test.gif", ims)
