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
        action = action / 20.
        reward = 0
        info = {}

        action = np.clip(action, -0.05, 0.05)
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
        action = action / (np.abs(action).max() + 1e-5)
        action = action / 20.

        reward = 0
        info = {}

        action = np.clip(action, -0.05, 0.05)
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
    def __init__(self, end=[0.7, -0.8], eps=0.01):
        self.env = gym.make("FetchPush-v1")
        self.target = np.array(end)

    def reset(self):
        self.env.reset()
        ob = self._get_obs()

        return ob

    def step(self, action):
        # Scale down action from range (-1, 1) to (-0.05, 0.05)
        reward = 0

        _, _, done, info = self.env.step(action)
        obs = self._get_obs()

        dist = np.abs(self.obs[:2] - self.target).sum()
        reward = -1 * dist

        if dist < self.eps:
            done = True
        else:
            done = done

        return observation, reward, done, info

    def seed(self, seed):
        np.random.seed(seed)

    def _get_obs(self):
        qpos = self.env.sim.data.qpos.copy()
        qvel = self.env.sim.data.qvel.copy()
        obs = np.concatenate([qpos[:2], qvel[:2]], axis=0)
        return obs
