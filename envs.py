import gym
import numpy as np

from gen_data import is_maze_valid

class Point(gym.Env):
	def __init__(self, start=[0.0, 0.0], end=[0.5, 0.5], eps=0.01):
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

		self.current = self.current + action
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

		temp = self.current + action
		if is_maze_valid(temp[None, :][0]):
			self.current = temp
		else:
			pass

		self.current = np.clip(self.current, -1, 1)
		observation = self.current

		if self.end[0] - self.eps <= self.current[0] <= self.end[0] + self.eps and self.end[1] - self.eps <= \
				self.current[0] <= self.end[1] + self.eps:
			done = True
		else:
			done = False



		return observation, reward, done, info