"""
A generic file to
(1) take a trained model for the specified environment
(2) run cond/no_cond benchmark
"""

import datetime
import os
import os.path as osp

import matplotlib as mpl
import tensorflow as tf
from baselines.logger import TensorBoardOutputFormat
from tensorflow.python.platform import flags

from traj_model import TrajInverseDynamics, TrajNetLatentFC

mpl.use('Agg')
import matplotlib.pyplot as plt
import torch
import numpy as np
from custom_adam import AdamOptimizer
# from render_utils import render_reach
import seaborn as sns

from envs import Point, Maze

sns.set()

# from inception import get_inception_score
# from fid import get_fid_score

torch.manual_seed(1)
FLAGS = flags.FLAGS

# Dataset Options
flags.DEFINE_string('type', 'random', 'random or past for initialization of new frame')
flags.DEFINE_string('datasource', 'point', 'point or maze')
flags.DEFINE_integer('batch_size', 256, 'Size of inputs')
flags.DEFINE_integer('data_workers', 6, 'Number of different data workers to load data in parallel')

# General Experiment Seittings
flags.DEFINE_string('logdir', 'cachedir', 'location where log of experiments will be stored')
flags.DEFINE_string('imgdir', 'rollout_images', 'location where image results of experiments will be stored')
flags.DEFINE_string('exp', 'default', 'name of experiments')
flags.DEFINE_integer('log_interval', 10, 'log outputs every so many batches')
flags.DEFINE_integer('save_interval', 1000, 'save outputs every so many batches')
flags.DEFINE_integer('test_interval', 1000, 'evaluate outputs every so many batches')
flags.DEFINE_integer('resume_iter', -1, 'iteration to resume training from')
flags.DEFINE_integer('epoch_num', 10, 'Number of Epochs to train on')
flags.DEFINE_float('lr', 1e-3, 'Learning for training')
flags.DEFINE_integer('seed', 0, 'Value of seed')

# Custom Experiments Settings
flags.DEFINE_integer('num_gpus', 1, 'number of gpus to train on')
flags.DEFINE_float('ml_coeff', 1.0, 'Coefficient to multiply maximum likelihood (descriminator coefficient)')
flags.DEFINE_float('l2_coeff', 1.0, 'Scale of regularization')

flags.DEFINE_integer('num_steps', 20, 'Steps of gradient descent for training')

# Architecture Settings
flags.DEFINE_bool('max_pool', False, 'Whether or not to use max pooling rather than strided convolutions')
flags.DEFINE_integer('num_filters', 64, 'number of filters for networks')
flags.DEFINE_bool('spec_norm', True, 'Whether to use spectral normalization on weights')
flags.DEFINE_bool('use_bias', True, 'Whether to use bias in convolution')
flags.DEFINE_integer('input_objects', 1, 'Number of objects to predict the trajectory of.')
flags.DEFINE_integer('latent_dim', 24, 'Number of dimension encoding state of object')
flags.DEFINE_integer('action_dim', 24, 'Number of dimension for encoding action of object')

# Custom EBM Architecture
flags.DEFINE_integer('total_frame', 2, 'Number of frames to train the energy model')
flags.DEFINE_bool('replay_batch', True, 'Whether to use a replay buffer for samples')
flags.DEFINE_bool('cond', False, 'Whether to condition on actions')
flags.DEFINE_bool('zero_kl', True, 'whether to make the kl be zero')
flags.DEFINE_integer('temperature', 1, 'Temperature for energy function')

# Projected gradient descent
flags.DEFINE_float('proj_norm', 0.00, 'Maximum change of input images')
flags.DEFINE_string('proj_norm_type', 'li', 'What type of ball for projection, only support l2 and li')

# Custom MCMC parameters
flags.DEFINE_float('step_lr', 1.0, 'Size of steps for gradient descent')
flags.DEFINE_bool('grad_free', False, 'instead of using gradient descent to generate latents, use DFO')
flags.DEFINE_integer('noise_sim', 20, 'Number of forward evolution steps to calculate')
flags.DEFINE_string('objective', 'cd', 'objective used to train EBM')

# Parameters for Planning
flags.DEFINE_integer('plan_steps', 10, 'Number of steps of planning')
flags.DEFINE_bool('seq_plan', False, 'Whether to use joint planning or sequential planning')

# Number of benchmark experiments
flags.DEFINE_integer('n_benchmark_exp', 0, 'Number of benchmark experiments')
flags.DEFINE_float('start1', 0.0, 'x_start, x')
flags.DEFINE_float('start2', 0.0, 'x_start, y')
flags.DEFINE_float('end1', 0.5, 'x_end, x')
flags.DEFINE_float('end2', 0.5, 'x_end, y')

FLAGS.batch_size *= FLAGS.num_gpus

# set_seed(FLAGS.seed)

if FLAGS.datasource == 'point':
	FLAGS.latent_dim = 2
	FLAGS.action_dim = 2
elif FLAGS.datasource == 'maze':
	FLAGS.latent_dim = 2
	FLAGS.action_dim = 2


def log_step_num_exp(d):
	import csv
	with open('get_avg_step_num_log.csv', mode='a+') as csv_file:
		fieldnames = ['ts', 'start', 'end', 'plan_steps', 'no_cond', 'step_num', 'exp', 'iter']
		writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
		writer.writerow(d)


def get_avg_step_num(target_vars, sess, env):
	n_exp = FLAGS.n_benchmark_exp
	cond = 'True' if FLAGS.cond else 'False'
	obs = env.reset()
	collected_trajs = []

	for i in range(n_exp):
		points = []
		while True:
			current_point = obs
			end_point = env.end
			X_START = target_vars['X_START']
			X_END = target_vars['X_END']
			X_PLAN = target_vars['X_PLAN']
			x_joint = target_vars['x_joint']
			output_actions = target_vars['actions']

			x_start = current_point[None, None, None, :]
			x_end = end_point[None, None, None, :]
			x_plan = np.random.uniform(-1, 1, (1, plan_steps, 1, 2))

			if not FLAGS.cond:
				x_joint, output_actions = sess.run([x_joint, output_actions],
				                                   {X_START: x_start, X_END: x_end, X_PLAN: x_plan})
			else:
				ACTION_PLAN = target_vars['ACTION_PLAN']
				actions = np.random.uniform(-0.05, 0.05, (1, plan_steps + 1, 2))
				x_joint, output_actions = sess.run([x_joint, output_actions],
				                                   {X_START: x_start, X_END: x_end,
				                                    X_PLAN: x_plan, ACTION_PLAN: actions})

			obs, _, done, _ = env.step(output_actions.squeeze()[0])
			print("obs", obs)
			print("actions", output_actions)
			points.append(output_actions)

			if done:
				break

		# log number of steps for each experiment
		ts = str(datetime.datetime.now())
		d = {'ts': ts,
		     'start': x_start,
		     'end': x_end,
		     'cond': cond,
		     'step_num': len(points),
		     'exp': FLAGS.exp,
		     'iter': FLAGS.resume_iter}
		log_step_num_exp(d)

		collected_trajs.append(np.array(points))

	lengths = []
	for traj in collected_trajs:
		plt.plot(traj[:, 0], traj[:, 1])
		lengths.append(traj.shape[0])

	average_length = sum(lengths) / len(lengths)

	imgdir = FLAGS.imgdir
	if not osp.exists(imgdir):
		os.makedirs(imgdir)
	timestamp = str(datetime.datetime.now())
	save_dir = osp.join(imgdir, 'benchmark_{}_{}_iter{}_{}.png'.format(FLAGS.n_benchmark_exp, FLAGS.exp,
	                                                                   FLAGS.resume_iter, timestamp))
	plt.savefig(save_dir)
	print("average number of steps:", average_length)


def construct_no_cond_plan_model(model, weights, X_PLAN, X_START, X_END, ACTION_LABEL):
	x_joint = tf.concat([X_START, X_PLAN, X_END], axis=1)
	steps = tf.constant(0)
	c = lambda i, x: tf.less(i, FLAGS.num_steps)

	idyn_model = TrajInverseDynamics()
	weights = idyn_model.construct_weights(scope="inverse_dynamics", weights=weights)

	def mcmc_step(counter, x_joint):
		cum_energies = 0

		# Code for doing joint sampling over all possible states
		# for i in range(FLAGS.plan_steps - FLAGS.total_frame + 3):
		#     cum_energy = model.forward(x_joint[:, i:i+FLAGS.total_frame], weights, action_label=ACTION_LABEL)
		#     cum_energies = cum_energies + cum_energy

		# cum_energies = tf.Print(cum_energies, [cum_energies])
		# x_grad = tf.gradients(cum_energies, [x_joint])[0]
		# x_joint = x_joint - FLAGS.step_lr * x_grad

		# Code for doing sampling for beginning to end and then from end to beginning
		cum_energies = []

		if FLAGS.seq_plan:
			for i in range(FLAGS.plan_steps - FLAGS.total_frame + 3):
				x_temp = x_joint[:, i:i + FLAGS.total_frame]
				x_temp = x_temp + tf.random_normal(tf.shape(x_temp), mean=0.0, stddev=0.1)
				cum_energy = model.forward(x_temp, weights, action_label=ACTION_LABEL)
				x_grad = tf.gradients(cum_energy, [x_temp])[0]
				x_new = x_joint[:, i:i + FLAGS.total_frame] - FLAGS.step_lr * tf.cast(counter,
				                                                                      tf.float32) / FLAGS.num_steps * x_grad

				x_joint = tf.concat([x_joint[:, :i], x_new, x_joint[:, i + FLAGS.total_frame:]], axis=1)

				cum_energies.append(cum_energy)

			x_joint = tf.concat([X_START, x_joint[:, 1:FLAGS.plan_steps + 1], X_END], axis=1)

			for i in range(FLAGS.plan_steps - FLAGS.total_frame + 2, -1):
				x_temp = x_joint[:, i:i + FLAGS.total_frame]
				x_temp = x_temp + tf.random_normal(tf.shape(x_temp), mean=0.0, stddev=0.1)
				cum_energy = model.forward(x_temp, weights, action_label=ACTION_LABEL)
				x_grad = tf.gradients(cum_energy, [x_temp])[0]

				x_new = x_joint[:, i:i + FLAGS.total_frame] - FLAGS.step_lr * tf.cast(counter,
				                                                                      tf.float32) / FLAGS.num_steps * x_grad
				x_joint = tf.concat([x_joint[:, :i], x_new, x_joint[:, i + FLAGS.total_frame:]], axis=1)

				cum_energies.append(cum_energy)

			cum_energies = tf.concat(cum_energies, axis=1)
		else:
			x_joint = x_joint + tf.random_normal(tf.shape(x_joint), mean=0.0, stddev=0.01)
			for i in range(FLAGS.plan_steps - FLAGS.total_frame + 3):
				x_temp = x_joint[:, i:i + FLAGS.total_frame]
				cum_energy = model.forward(x_temp, weights, action_label=ACTION_LABEL)
				cum_energies.append(cum_energy)

			cum_energies = tf.reduce_sum(tf.concat(cum_energies, axis=1), axis=1)
			x_grad = tf.gradients(cum_energies, [x_joint])[0]
			x_joint = x_joint - FLAGS.step_lr * tf.cast(counter, tf.float32) / FLAGS.num_steps * x_grad

		# Reset the start and end states to be previous values
		x_joint = tf.concat([X_START, x_joint[:, 1:FLAGS.plan_steps + 1], X_END], axis=1)
		counter = counter + 1

		# counter = tf.Print(counter,
		#                    [tf.reduce_mean(cum_energies), tf.reduce_max(cum_energies), tf.reduce_min(cum_energies)])
		x_joint = tf.clip_by_value(x_joint, -1.0, 1.0)

		return counter, x_joint

	steps, x_joint = tf.while_loop(c, mcmc_step, (steps, x_joint))

	actions = idyn_model.forward(x_joint[:, 0:2], weights)
	target_vars = {}

	target_vars['actions'] = actions
	target_vars['x_joint'] = x_joint
	target_vars['X_START'] = X_START
	target_vars['X_END'] = X_END
	target_vars['ACTION_LABEL'] = ACTION_LABEL
	target_vars['X_PLAN'] = X_PLAN

	return target_vars


def construct_cond_plan_model(model, weights, X_PLAN, X_START, X_END, ACTION_PLAN):
	actions = ACTION_PLAN
	x_joint = tf.concat([X_START, X_PLAN, X_END], axis=1)
	steps = tf.constant(0)
	c = lambda i, x, y: tf.less(i, FLAGS.num_steps)

	def mcmc_step(counter, x_joint, actions):
		actions = actions + tf.random_normal(tf.shape(actions), mean=0.0, stddev=0.01)
		x_joint = x_joint + tf.random_normal(tf.shape(x_joint), mean=0.0, stddev=0.01)
		cum_energies = 0
		for i in range(FLAGS.plan_steps - FLAGS.total_frame + 3):
			print(x_joint[:, i:i + FLAGS.total_frame].get_shape())
			cum_energy = model.forward(x_joint[:, i:i + FLAGS.total_frame], weights, action_label=actions[:, i])
			cum_energies = cum_energies + cum_energy

		# cum_energies = tf.Print(cum_energies, [cum_energies])

		x_grad = tf.gradients(cum_energies, [x_joint])[0]
		x_joint = x_joint - FLAGS.step_lr * x_grad
		x_joint = tf.concat([X_START, x_joint[:, 1:FLAGS.plan_steps + 1], X_END], axis=1)
		x_joint = tf.clip_by_value(x_joint, -1.0, 1.0)

		action_grad = tf.gradients(cum_energies, [actions])[0]
		actions = actions - FLAGS.step_lr * action_grad
		actions = tf.clip_by_value(actions, -0.05, 0.05)

		counter = counter + 1

		return counter, x_joint, actions

	steps, x_joint, actions = tf.while_loop(c, mcmc_step, (steps, x_joint, actions))
	target_vars = {}
	target_vars['x_joint'] = x_joint
	target_vars['actions'] = actions
	target_vars['X_START'] = X_START
	target_vars['X_END'] = X_END
	target_vars['X_PLAN'] = X_PLAN
	target_vars['ACTION_PLAN'] = ACTION_PLAN

	return target_vars


def construct_model(model, weights, X_NOISE, X, ACTION_LABEL, LR, optimizer):
	target_vars = {}
	x_mods = []

	energy_pos = model.forward(X, weights, action_label=ACTION_LABEL)
	energy_noise = energy_start = model.forward(X_NOISE, weights, reuse=True, stop_at_grad=True,
	                                            action_label=ACTION_LABEL)

	print("Building graph...")
	x_mod = X_NOISE

	x_grads = []
	x_ees = []
	energy_negs = [energy_noise]
	loss_energys = []

	steps = tf.constant(0)
	c = lambda i, x: tf.less(i, FLAGS.num_steps)

	def mcmc_step(counter, x_mod):
		if FLAGS.grad_free:
			x_mod_neg = tf.tile(tf.expand_dims(x_mod, axis=1), (1, FLAGS.noise_sim, 1, 1, 1))
			x_mod_neg_shape = tf.shape(x_mod_neg)

			# User energies to set movement speed in derivative free optimization
			energy_noise = FLAGS.temperature * model.forward(x_mod, weights, action_label=ACTION_LABEL, reuse=True)
			energy_noise = tf.reshape(energy_noise, (x_mod_neg_shape[0], 1, 1, 1, 1))
			x_noise = tf.random_normal(x_mod_neg_shape, mean=0.0, stddev=0.05)
			x_mod_stack = x_mod_neg = x_mod_neg + x_noise
			x_mod_neg = tf.reshape(x_mod_neg, (
				x_mod_neg_shape[0] * x_mod_neg_shape[1], x_mod_neg_shape[2], x_mod_neg_shape[3], x_mod_neg_shape[4]))

			if ACTION_LABEL is not None:
				action_label_tile = tf.reshape(tf.tile(tf.expand_dims(ACTION_LABEL, dim=1), (1, FLAGS.noise_sim, 1)),
				                               (FLAGS.batch_size * FLAGS.noise_sim, -1))
			else:
				action_label_tile = None

			energy_noise = -FLAGS.temperature * model.forward(x_mod_neg, weights, action_label=action_label_tile,
			                                                  reuse=True)
			energy_noise = tf.reshape(energy_noise, (x_mod_neg_shape[0], FLAGS.noise_sim))
			energy_wt = tf.nn.softmax(energy_noise, axis=1)
			energy_wt = tf.reshape(energy_wt, (x_mod_neg_shape[0], FLAGS.noise_sim, 1, 1, 1))
			loss_energy_wt = tf.reshape(energy_wt, (-1, 1))
			loss_energy_noise = tf.reshape(energy_noise, (-1, 1))
			x_mod = tf.reduce_sum(energy_wt * x_mod_stack, axis=1)
			loss_energy_neg = loss_energy_noise * loss_energy_wt
		else:
			x_mod = x_mod + tf.random_normal(tf.shape(x_mod), mean=0.0, stddev=0.01)
			energy_noise = model.forward(x_mod, weights, action_label=ACTION_LABEL, reuse=True, stop_at_grad=True)
			lr = FLAGS.step_lr
			x_grad = tf.gradients(FLAGS.temperature * energy_noise, [x_mod])[0]
			x_mod = x_mod - lr * x_grad

		x_mod = tf.clip_by_value(x_mod, -1.2, 1.2)

		counter = counter + 1

		return counter, x_mod

	steps, x_mod = tf.while_loop(c, mcmc_step, (steps, x_mod))

	target_vars['x_mod'] = x_mod
	temp = FLAGS.temperature

	loss_energy = temp * model.forward(x_mod, weights, reuse=True, action_label=ACTION_LABEL, stop_grad=True)
	x_mod = tf.stop_gradient(x_mod)

	energy_neg = model.forward(x_mod, weights, action_label=ACTION_LABEL, reuse=True)
	x_grad = tf.gradients(FLAGS.temperature * energy_neg, [x_mod])[0]
	x_off = tf.reduce_mean(tf.square(x_mod - X))

	if FLAGS.train:
		if FLAGS.objective == 'logsumexp':
			pos_term = temp * energy_pos
			energy_neg_reduced = (energy_neg - tf.reduce_min(energy_neg))
			coeff = tf.stop_gradient(tf.exp(-temp * energy_neg_reduced))
			norm_constant = tf.stop_gradient(tf.reduce_sum(coeff)) + 1e-4
			pos_loss = tf.reduce_mean(temp * energy_pos)
			neg_loss = coeff * (-1 * temp * energy_neg) / norm_constant
			loss_ml = FLAGS.ml_coeff * (pos_loss + tf.reduce_sum(neg_loss))
		elif FLAGS.objective == 'cd':
			pos_loss = tf.reduce_mean(temp * energy_pos)
			neg_loss = -tf.reduce_mean(temp * energy_neg)
			loss_ml = FLAGS.ml_coeff * (pos_loss + tf.reduce_sum(neg_loss))
		elif FLAGS.objective == 'softplus':
			loss_ml = FLAGS.ml_coeff * \
			          tf.nn.softplus(temp * (energy_pos - energy_neg))

	loss_total = tf.reduce_mean(loss_ml)

	if not FLAGS.zero_kl:
		loss_total = loss_total + tf.reduce_mean(loss_energy)

	loss_total = loss_total + \
	             FLAGS.l2_coeff * (tf.reduce_mean(tf.square(energy_pos)) + tf.reduce_mean(tf.square((energy_neg))))

	if FLAGS.train:
		print("Started gradient computation...")
		gvs = optimizer.compute_gradients(loss_total)
		gvs = [(k, v) for (k, v) in gvs if k is not None]
		print("Applying gradients...")
		grads, vs = zip(*gvs)

		def filter_grad(g, v):
			return tf.clip_by_value(g, -1e5, 1e5)

		capped_gvs = [(filter_grad(grad, var), var) for grad, var in gvs]
		gvs = capped_gvs
		train_op = optimizer.apply_gradients(gvs)
		target_vars['train_op'] = train_op

	print("Finished applying gradients.")
	target_vars['loss_ml'] = loss_ml
	target_vars['total_loss'] = loss_total
	target_vars['gvs'] = gvs
	target_vars['loss_energy'] = loss_energy
	target_vars['weights'] = weights
	target_vars['X'] = X
	target_vars['X_NOISE'] = X_NOISE
	target_vars['energy_pos'] = energy_pos
	target_vars['energy_neg'] = energy_neg
	target_vars['x_grad'] = x_grad
	target_vars['x_mod'] = x_mod
	target_vars['x_off'] = x_off
	target_vars['temp'] = temp
	target_vars['lr'] = LR
	target_vars['ACTION_LABEL'] = ACTION_LABEL

	return target_vars


def main():
	logdir = osp.join(FLAGS.logdir, FLAGS.exp)
	if not osp.exists(logdir):
		os.makedirs(logdir)
	logger = TensorBoardOutputFormat(logdir)

	# Only know the setting for omniglot, not sure about others
	batch_size = FLAGS.batch_size

	if FLAGS.datasource == 'point' or FLAGS.datasource == 'maze':
		model = TrajNetLatentFC(dim_input=FLAGS.total_frame)
		X_NOISE = tf.placeholder(shape=(None, FLAGS.total_frame, FLAGS.input_objects, FLAGS.latent_dim),
		                         dtype=tf.float32)
		X = tf.placeholder(shape=(None, FLAGS.total_frame, FLAGS.input_objects, FLAGS.latent_dim), dtype=tf.float32)
		ACTION_LABEL = tf.placeholder(shape=(None, 2), dtype=tf.float32)
		ACTION_PLAN = tf.placeholder(shape=(None, FLAGS.plan_steps + 1, 2), dtype=tf.float32)

		X_START = tf.placeholder(shape=(None, 1, FLAGS.input_objects, FLAGS.latent_dim), dtype=tf.float32)
		X_PLAN = tf.placeholder(shape=(None, FLAGS.plan_steps, FLAGS.input_objects, FLAGS.latent_dim), dtype=tf.float32)
		X_END = tf.placeholder(shape=(None, 1, FLAGS.input_objects, FLAGS.latent_dim), dtype=tf.float32)

	if not FLAGS.cond:
		ACTION_LABEL = None

	weights = model.construct_weights(action_size=FLAGS.action_dim)
	LR = tf.placeholder(tf.float32, [])
	optimizer = AdamOptimizer(LR, beta1=0.0, beta2=0.999)

	if not FLAGS.cond:
		target_vars = construct_no_cond_plan_model(model, weights, X_PLAN, X_START, X_END, ACTION_LABEL)
	else:
		target_vars = construct_cond_plan_model(model, weights, X_PLAN, X_START, X_END, ACTION_PLAN)

	sess = tf.InteractiveSession()
	saver = loader = tf.train.Saver(max_to_keep=10, keep_checkpoint_every_n_hours=2)

	tf.global_variables_initializer().run()
	print("Initializing variables...")

	resume_itr = 0

	if FLAGS.resume_iter != -1 or not FLAGS.train:
		model_file = osp.join(logdir, 'model_{}'.format(FLAGS.resume_iter))
		resume_itr = FLAGS.resume_iter
		saver.restore(sess, model_file)

	if FLAGS.datasource == 'point':
		env = Point()
	elif FLAGS.datasource == 'maze':
		env = Maze()
	else:
		raise KeyError

	get_avg_step_num(target_vars, sess, env)


if __name__ == "__main__":
	main()
