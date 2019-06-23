import numpy as np
from trajopt.envs.reacher_env import Reacher7DOFEnv


class ContinualReacher7DOFEnv(Reacher7DOFEnv):
    def __init__(self):
        super().__init__()
        self.env_name = 'continual_reacher_7dof'

    def _step(self, a):
        self.do_simulation(a, self.frame_skip)
        hand_pos = self.data.site_xpos[self.hand_sid]
        target_pos = self.data.site_xpos[self.target_sid]
        dist = np.linalg.norm(hand_pos-target_pos)
        reward = - 10.0 * dist # - 0.25 * np.linalg.norm(self.data.qvel)
        ob = self._get_obs()

        # continual components
        self.env_timestep += 1
        if self.env_timestep % 50 == 0 and self.env_timestep > 0 and self.real_step is True:
            self.target_reset()

        if self.env_timestep > 100:
            done = True
        else:
            done = False

        return ob, reward, done, self.get_env_infos()

    def reset(self):
        self.target_reset()
        self.reset_model()
        ob = self._get_obs()

        return ob
