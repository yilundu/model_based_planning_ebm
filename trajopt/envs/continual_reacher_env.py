import numpy as np
from trajopt.envs.reacher_env import Reacher7DOFEnv


class ContinualReacher7DOFEnv(Reacher7DOFEnv):
    def __init__(self, train=True):
        # self.count = np.random.randint(-10, 10)
        self.count = 0
        super().__init__()
        self.env_name = 'continual_reacher_7dof'
        self.train = train

    def _step(self, a):
        self.do_simulation(a, self.frame_skip)
        hand_pos = self.data.site_xpos[self.hand_sid]
        target_pos = self.model.site_pos[self.target_sid]
        dist = np.linalg.norm(hand_pos-target_pos)

        # target_pos = [ 0.00068051,  0.01088833, -0.0469095 , -0.02827646,  0.02504256, -0.07271489,  0.04032468]
        # dist = np.linalg.norm(self.sim.data.qpos-target_pos)
        reward = - 10.0 * dist # - 0.25 * np.linalg.norm(self.data.qvel)
        ob = self._get_obs()

        # continual components
        self.env_timestep += 1
        # if self.env_timestep % 50 == 0 and self.env_timestep > 0 and self.real_step is True:
        #     self.target_reset()


        if (self.env_timestep % (100 + self.count) == 0 or (dist < 0.10 and not self.train)) and not (target_pos.sum() == 0):
            done = True
        else:
            done = False

        return ob, reward, done, self.get_env_infos()

    def reset(self):
        # self.target_reset()
        self.reset_model()

        if not self.train:
            self.count = 100

        self.model.site_pos[self.target_sid] = [0.1, 0.1, 0.1]

        observation, _reward, done, _info = self._step(np.zeros(7))
        ob = self._get_obs()

        return ob
