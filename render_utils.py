from gpr.envs.dactyl_reach import make_env
import numpy as np
from gpr.envs import dactyl_locked


dactyl_reach = make_env(randomize=False, her_support=True)
dactyl_locked = dactyl_locked.make_env(randomize=False, her_support=True)

def render_reach(obs, task, mean, std, im_size=200):
    ims = []

    if task == 'hand':
        env = dactyl_locked
    elif task == 'fetch':
        env = dactyl_reach

    if task == 'hand':
        # obs[:, :, 24:31] = obs[:, :, 24:31] / 10.
        # obs /= 10
        pass

    for i in range(obs.shape[0]):
        sim = env.unwrapped.sim

        sim.data.qpos[:] = obs[i, 0] * std + mean
        sim.forward()
        # im = dactyl_reach._get_viewer().read_pixels(im_size, im_size, depth=False)
        im = sim.render(im_size, im_size)
        ims.append(im)
        sim.reset()

    im = np.stack(ims, axis=0)

    return im
