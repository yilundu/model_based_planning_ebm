import gym
import numpy as np

from gen_data import is_maze_valid


class Point(gym.Env):
    def __init__(self, start=[0.0, 0.0], end=[0.5, 0.5], eps=0.05, obstacle=None, random_starts=False):
        self.start = np.array(start)
        self.end = np.array(end)
        self.current = np.array(start)
        self.obstacle = obstacle  # obstacle should be a size 4 array specifying top left and bottom right
        self.random_starts = random_starts

        self.eps = eps

    def reset(self):
        if self.random_starts:
            self.current = np.random.uniform(-1, 1, (2))
        else:
            self.current = self.start
        # print("Reset ", self.current)
        return self.current

    def is_step_valid(self, pos):
        top, left, bottom, right = self.obstacle
        if left <= pos[0] <= right and bottom <= pos[1] <= top:
            return False
        else:
            return True

    def step(self, action):
        # Scale down action from range (-1, 1) to (-0.05, 0.05)
        action = action / max(np.abs(action).max(), 1)
        action = action / 20.
        reward = 0
        info = {}

        # action = np.clip(action, -0.05, 0.05)
        temp = self.current + action
        if self.obstacle is not None:
            if self.is_step_valid(temp):
                self.current = temp
            else:
                pass
        else:
            self.current = temp

        self.current = np.clip(self.current, -1, 1)
        observation = self.current

        dist = np.abs(self.current - self.end).sum()
        reward = -1 * dist

        if dist < self.eps:
            done = True
        else:
            done = False

        return observation, reward, done, info

    def seed(self, seed):
        np.random.seed(seed)


class Maze(gym.Env):
    def __init__(self, start=[0.1, 0.0], end=[0.7, -0.8], eps=0.01, obstacle=[0.1, 0.1], random_starts=False):
        self.start = np.array(start)
        self.end = np.array(end)
        self.current = np.array(start)
        self.random_starts = random_starts

        self.eps = eps

    def reset(self):
        if self.random_starts:
            self.current = np.random.uniform(-1, 1, (2))
            if not is_maze_valid(self.current[None, :])[0]:
                self.reset()
        else:
            self.current = self.start

        print("Reset: ", self.current)

        return self.current

    def step(self, action):
        # Scale down action from range (-1, 1) to (-0.05, 0.05)
        action = action / max(np.abs(action).max(), 1)
        action = action / 20.

        reward = 0
        info = {}

        # action = np.clip(action, -0.05, 0.05)
        temp = self.current + action
        if is_maze_valid(temp[None, :])[0]:
            self.current = temp
        else:
            pass

        self.current = np.clip(self.current, -1, 1)
        observation = self.current

        dist = np.abs(self.current - self.end).sum()
        reward = -1 * dist

        if dist < self.eps:
            done = True
        else:
            done = False

        return observation, reward, done, info

    def seed(self, seed):
        np.random.seed(seed)


class Reacher(gym.Env):
    def __init__(self, end=[0.7, 0.5], eps=0.01, pretrain_eval=False):
        self.env = gym.make("Reacher-v2")

        # Internal variable for computing reward
        self.target = np.array(end)

        # Visible variable indicating the goal state
        self.end = self.target

        if pretrain_eval:
            self.cross_preprocess(end)

        self.eps = eps
        self.pretrain_eval = pretrain_eval

        self.mean = np.array([-0.00414585, -0.00524412, -0.0067701 , -0.00752602])
        self.std = np.array([3.51472521, 1.72293722, 8.62294938, 6.39224953])

    def cross_preprocess(self, end):
        end[:2] = end[:2] * np.pi + np.pi
        end[2:4] = end[2:4] * 10.0

        target = (end - self.mean) / self.std
        self.end = end

    def reset(self):
        self.env.reset()
        ob = self._get_obs()

        return ob

    def step(self, action):
        # Scale down action from range (-1, 1) to (-0.05, 0.05)
        reward = 0

        _, _, done, info = self.env.step(action)
        obs = self._get_obs()

        reward = self.reward()
        dist = -1 * reward

        if dist < self.eps:
            done = True
        else:
            done = done

        return obs, reward, done, info

    def seed(self, seed):
        np.random.seed(seed)

    def reward(self):
        qpos = self.env.unwrapped.sim.data.qpos.copy()
        qvel = self.env.unwrapped.sim.data.qvel.copy()
        obs = np.concatenate([qpos[:2], qvel[:2]], axis=0)
        obs[:2] = ((obs[:2] % (2 * np.pi)) - np.pi) / np.pi
        obs[2:4] = obs[2:4] / 10.0

        dist = np.minimum(np.minimum(np.abs(obs[:2] - self.target), np.abs(self.target + 2 - obs[:2])), np.abs(self.target - 2 -obs[:2])).sum()
        reward = -1 * dist

        return reward

    def _get_obs(self):
        qpos = self.env.unwrapped.sim.data.qpos.copy()
        qvel = self.env.unwrapped.sim.data.qvel.copy()
        obs = np.concatenate([qpos[:2], qvel[:2]], axis=0)

        if self.pretrain_eval:
            obs = (obs - mean) / std
        else:
            obs[:2] = ((obs[:2] % (2 * np.pi)) - np.pi) / np.pi
            obs[2:4] = obs[2:4] / 10.0

        return obs
