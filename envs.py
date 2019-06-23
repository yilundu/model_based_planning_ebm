from collections import namedtuple

import gym
import numpy as np

from utils import is_maze_valid

Rectangle = namedtuple('Rectangle', 'xmin ymin xmax ymax')


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

    def is_overlapping(self, a, b):  # returns None if rectangles don't intersect
        dx = min(a.xmax, b.xmax) - max(a.xmin, b.xmin)
        dy = min(a.ymax, b.ymax) - max(a.ymin, b.ymin)
        if (dx >= 0) and (dy >= 0):
            return True
        else:
            return False

    def is_step_valid(self, curr_pos, next_pos):
        top, left, bottom, right = self.obstacle

        ra = Rectangle(left, bottom, right, top)
        rb = Rectangle(min(curr_pos[0], next_pos[0]),
                       min(curr_pos[1], next_pos[1]),
                       max(curr_pos[0], next_pos[0]),
                       max(curr_pos[1], next_pos[1]))

        if self.is_overlapping(ra, rb):
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
            if self.is_step_valid(self.current, temp):
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


class Ball(gym.Env):
    def __init__(self, start=[0.1, 0.1], end=[1.0, 1.0], a=[0.0, 0.0], eps=0.05, cor=1.0, random_starts=False, phy_type=None):
        self.start = np.array(start)
        self.end = np.array(end)
        self.random_starts = random_starts

        # range of container
        self.low = 0.0
        self.high = 1.0

        # initial velocity
        self.v = np.array([0.0, 0.0])

        # random acceleration; a physical parameter to be inferred
        self.a = np.array(a)

        # coefficient of restitution, delta v'/ delta v; a physical parameter to be inferred
        self.cor = cor

        self.phy_type = phy_type
        if phy_type == 'a':
            self.start = np.concatenate((self.start, self.a))
            self.end = np.concatenate((self.end, self.a))
        elif phy_type == 'cor':
            self.start = np.append(self.start, self.cor)
            self.end = np.append(self.end, self.cor)

        self.current = self.start

        self.eps = eps

    def reset(self):
        if self.random_starts:
            self.current[:2] = np.random.uniform(0.0, 1.0, (2))
        else:
            self.current = self.start

        print("Reset ball position to: ", self.current)

        return self.current

    def collide(self, current, action):
        # free movement for one time step

        # print("Start position", current)
        # print("Start velocity", action)

        action += self.a   # add random force (~air resistance/wind)

        end = []

        # check for collisions with container's wall
        for i, vi in list(zip(current, action)):
            if i <= self.low:
                i = -i
                vi = -vi
                vi *= self.cor
            elif i >= self.high:
                i = 2 * self.high - i
                vi = -vi
                vi *= self.cor
            else:
                pass

            end.append((i, vi))

        # unzip
        current, v = zip(*end)
        current = np.array(current)
        v = np.array(v)

        # print("Final position", current)
        # print("Final velocity", v)

        self.v = v

        # return updated position
        return current

    def move(self):
        # step without action
        start = self.current

        self.current, self.v = self.collide(self.current, self.v)

        # calculate the "action" (i.e. reverse dynamics)
        action = self.current - start

        return action

    def step(self, action):
        # Scale down action from range (-1, 1) to (-0.05, 0.05)
        action = action / max(np.abs(action).max(), 1)
        action = action / 20.
        reward = 0
        info = {}

        # action = np.clip(action, -0.05, 0.05)
        self.current[:2] = self.collide(self.current[:2], action)

        dist = np.abs(self.current[:2] - self.end[:2]).sum()
        reward = -1 * dist

        if dist < self.eps:
            done = True
        else:
            done = False

        observation = self.current

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
