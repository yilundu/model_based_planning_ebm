import gym
import numpy as np

from gen_data import is_maze_valid


class Point(gym.Env):
    def __init__(self, start=[0.0, 0.0], end=[0.5, 0.5], eps=0.01, obstacle=None):
        self.start = np.array(start)
        self.end = np.array(end)
        self.current = np.array(start)
        self.obstacle = obstacle  # obstacle should be a size 4 array specifying top left and bottom right

        self.eps = eps

    def reset(self):
        self.current = self.start
        return self.current

    def is_step_valid(self, pos):
        top, left, bottom, right = self.obstacle
        if left <= pos[0] <= right and bottom <= pos[1] <= top:
            return False
        else:
            return True

    def step(self, action):
        reward = 0
        info = {}

        action = np.clip(action, -0.05, 0.05)
        print("Actions: ", action)
        temp = self.current + action
        if self.obstacle != None:
            if self.is_step_valid(temp):
                self.current = temp
            else:
                pass
        else:
            self.current = temp

        self.current = np.clip(self.current, -1, 1)
        observation = self.current

        if np.abs(self.current - self.end).sum() < self.eps:
            done = True
        else:
            done = False

        return observation, reward, done, info


class Maze(gym.Env):
    def __init__(self, start=[0.1, 0.0], end=[0.7, -0.8], eps=0.01):
        self.start = np.array(start)
        self.end = np.array(end)
        self.current = np.array(start)

        self.eps = eps

    def reset(self):
        self.current = self.start
        return self.current

    def step(self, action):
        reward = 0
        info = {}

        action = np.clip(action, -0.05, 0.05)
        temp = self.current + action
        if is_maze_valid(temp[None, :][0]):
            self.current = temp
        else:
            pass

        self.current = np.clip(self.current, -1, 1)
        observation = self.current

        if np.abs(self.current - self.end).sum() < self.eps:
            done = True
        else:
            done = False

        return observation, reward, done, info
