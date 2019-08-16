import time
import os
import os.path as osp
import random
import datetime

import gym
import imageio
import matplotlib as mpl
import matplotlib.patches as patches
import tensorflow as tf
from baselines.bench import Monitor
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.logger import TensorBoardOutputFormat
from tensorflow.python.platform import flags
from scipy.linalg import toeplitz
from tensorflow.contrib.distributions import MultivariateNormalFullCovariance

from traj_model import TrajFFDynamics, TrajInverseDynamics, TrajNetLatentFC

mpl.use('Agg')
import matplotlib.pyplot as plt
# import torch
import numpy as np
from itertools import product
from custom_adam import AdamOptimizer
# from render_utils import render_reach
from utils import ReplayBuffer, log_step_num_exp, parse_valid_obs, safemean
from traj_utils import continual_reacher_inverse_dynamics, process_trajectory, linear_reacher_inverse_dynamics, update_linear_weight

import seaborn as sns
import pandas as pd

from trajopt.envs.continual_reacher_env import ContinualReacher7DOFEnv
from envs import Point, Maze, Reacher, Ball
from tqdm import tqdm

sns.set()
plt.rcParams["font.family"] = "Times New Roman"

# from inception import get_inception_score
# from fid import get_fid_score

# torch.manual_seed(1)
FLAGS = flags.FLAGS

# Dataset Options
flags.DEFINE_string('datasource', 'point', 'point or maze or reacher or phy_a or phy_cor or collision')
flags.DEFINE_integer('batch_size', 256, 'Size of inputs')
flags.DEFINE_bool('single_task', False, 'whether to train on a single task')

flags.DEFINE_bool('pretrain_eval', False,
                  'either evaluate from pretraining dataset or from online dataset (since there are discrepancies)')

# General Experiment Seittings
flags.DEFINE_string('datadir', './data/', 'location where data is stored')
flags.DEFINE_string('logdir', 'cachedir', 'location where log of experiments will be stored')
flags.DEFINE_string('imgdir', 'rollout_images', 'location where image results of experiments will be stored')
flags.DEFINE_string('exp', 'default', 'name of experiments')
flags.DEFINE_integer('log_interval', 10, 'log outputs every so many batches')
flags.DEFINE_integer('save_interval', 200, 'save outputs every so many batches')
flags.DEFINE_integer('resume_iter', -1, 'iteration to resume training from')
flags.DEFINE_bool('train', True, 'whether to train or test')
flags.DEFINE_bool('debug', False, 'debug what is going on for conditional models')
flags.DEFINE_integer('epoch_num', 10, 'Number of Epochs to train on')
flags.DEFINE_float('lr', 1e-3, 'Learning for training')
flags.DEFINE_bool('heatmap', False, 'Visualize the heatmap in environments')

# Custom Experiments Settings
flags.DEFINE_integer('num_gpus', 1, 'number of gpus to train on')
flags.DEFINE_float('ml_coeff', 1.0, 'Coefficient to multiply maximum likelihood (descriminator coefficient)')
flags.DEFINE_float('l2_coeff', 1.0, 'Scale of regularization')

flags.DEFINE_integer('num_steps', 0, 'Steps of gradient descent for training ebm')
flags.DEFINE_integer('num_plan_steps', 10, 'Steps of gradient descent for generating plans')
flags.DEFINE_bool('random_shuffle', True, 'whether to shuffle input data')

# Architecture Settings
flags.DEFINE_integer('input_objects', 1, 'Number of objects to predict the trajectory of.')
flags.DEFINE_integer('latent_dim', 24, 'Number of dimension encoding state of object')
flags.DEFINE_integer('action_dim', 24, 'Number of dimension for encoding action of object')

# Custom EBM Architecture
flags.DEFINE_integer('total_frame', 2, 'Number of frames to train the energy model')
flags.DEFINE_bool('replay_batch', True, 'Whether to use a replay buffer for samples')
flags.DEFINE_bool('cond', False, 'Whether to condition on actions')
flags.DEFINE_bool('zero_kl', True, 'whether to make the kl be zero')
flags.DEFINE_bool('spec_norm', False, 'spectral norm for the networks')
flags.DEFINE_integer('temperature', 1, 'Temperature for energy function')

# Custom MCMC parameters
flags.DEFINE_float('step_lr', 1.0, 'Size of steps for gradient descent')
flags.DEFINE_bool('grad_free', False, 'instead of using gradient descent to generate latents, use DFO')
flags.DEFINE_integer('noise_sim', 40, 'Number of forward evolution steps to calculate')
flags.DEFINE_string('objective', 'cd', 'objective used to train EBM')

# Parameters for Planning 
flags.DEFINE_integer('plan_steps', 10, 'Number of steps of planning')
flags.DEFINE_bool('anneal', False, 'Whether to use simulated annealing for sampling')
flags.DEFINE_bool('mppi', True, 'whether to use MPPI for planning instead of Langevin dynamics')

# Parameters for benchmark experiments
flags.DEFINE_integer('n_benchmark_exp', 0, 'Number of benchmark experiments')
flags.DEFINE_float('start1', 0.0, 'x_start, x')
flags.DEFINE_float('start2', 0.0, 'x_start, y')
flags.DEFINE_float('end1', 0.5, 'x_end, x')
flags.DEFINE_float('end2', 0.5, 'x_end, y')
flags.DEFINE_float('eps', 0.01, 'epsilon for done condition')
flags.DEFINE_list('obstacle', [0.5, 0.1, 0.1, 0.5],
                  'a size 4 array specifying top left and bottom right, e.g. [0.25, 0.35, 0.3, 0.3]')
flags.DEFINE_bool('score_reward', True, 'Always score the reward')
flags.DEFINE_bool('log_traj', False, 'Don"t log trajectory by default')

# Additional constraints
flags.DEFINE_bool('constraint_vel', True, 'A distance constraint between each subsequent state')
flags.DEFINE_bool('constraint_goal', True, 'A distance constraint between current state and goal state')
flags.DEFINE_bool('constraint_accel', True, 'An acceleration constraint on trajectory')

flags.DEFINE_bool('gt_inverse_dynamics', True, 'if True, use GT dynamics; if False, train a inverse dynamics model')
flags.DEFINE_bool('linear_inverse_dynamics', False, 'do inverse dynamics through a linear regression')

# Constraints for RL training only
flags.DEFINE_bool('random_action', False, 'instead of using the modeling to predict actions, use random actions instead')

# use FF to train forward prediction rather than EBM
flags.DEFINE_bool('ff_model', False, 'Run action conditional with a deterministic FF network')

# Hyperparameters for RL training
flags.DEFINE_bool('rl_train', True, 'If true, run rl_train() instead of train()')
flags.DEFINE_integer('seed', 1, 'Seed to use when running environments')
flags.DEFINE_integer('nsteps', int(1e7), 'Total of steps of the environment to run')
flags.DEFINE_integer('num_env', 16, 'Number of different environments to run in parallel')
flags.DEFINE_bool('smooth_path', True, 'Initialize states to be smooth')
flags.DEFINE_bool('opt_low', False, 'using std as part of exploration')

flags.DEFINE_integer('n_exp', 1, 'Number of tests run for training and testing')
flags.DEFINE_float('v_coeff', 0.0, 'velocity coefficient')
flags.DEFINE_float('g_coeff', 1.0, 'goal coefficient')
flags.DEFINE_float('l_coeff', 0.0, 'l2 to last state coefficient')
flags.DEFINE_float('a_coeff', 1.0, 'acceleration coefficient')
flags.DEFINE_float('traj_scale', 1.0, 'scaling on smooth trajectories')
flags.DEFINE_bool('adaptive_sample', True, 'whether to adaptively sample')
flags.DEFINE_bool('energy_heatmap', False, 'generate energy heatmap')

# Custom stats for continual reacher
flags.DEFINE_bool('end_effector_stat', False, 'Track positions of the finger')
flags.DEFINE_bool('record_reacher_data', False, 'Record the data seen by the reacher')
flags.DEFINE_bool('continual_reacher_model', False, 'Use data from the continual_reacher_model')
flags.DEFINE_bool('render_image', False, 'render images of trajectories for continual_reacher')

# If True, estimate physical params from latent variable
flags.DEFINE_bool('phy_latent', True, 'If True, estimate physical params from latent variable')
flags.DEFINE_bool('eval_collision', False, 'If evaluate collision between obstacles')

FLAGS.batch_size *= FLAGS.num_gpus

if FLAGS.datasource == 'point':
    FLAGS.latent_dim = 2
    FLAGS.action_dim = 2
elif FLAGS.datasource == 'maze':
    FLAGS.latent_dim = 2
    FLAGS.action_dim = 2
elif FLAGS.datasource == 'reacher':
    FLAGS.latent_dim = 4
    FLAGS.action_dim = 2
elif FLAGS.datasource == "continual_reacher":
    FLAGS.latent_dim = 20
    FLAGS.action_dim = 7
elif FLAGS.datasource == 'phy_a':
    FLAGS.latent_dim = 4   # s1, s2, phy_param
    FLAGS.action_dim = 2
elif FLAGS.datasource == 'phy_cor':
    FLAGS.latent_dim = 3  # s1, s2, phy_param
    FLAGS.action_dim = 2
elif FLAGS.datasource == 'collision':
    FLAGS.latent_dim = 6  # s1, s2, phy_param
    FLAGS.input_objects = 8

# if FLAGS.datasource == "reacher" or FLAGS.datasource == "continual_reacher":
#     FLAGS.gt_inverse_dynamics = True


mvn = None
def smooth_trajectory_tf(scale, batch_size=None):
    if batch_size is None:
        batch_size = FLAGS.num_env
    global mvn
    if mvn is None:
        n = FLAGS.plan_steps
        matrix = toeplitz([1, -2 , 1] + [0]*(n-1), [1] + [0] * (n-1))
        matrix[-2:] = 0
        matrix[-1, -2:] = [1, -1]

        # matrix[:2] = 0
        # matrix[0, 0:2] = [1, -1]
        # matrix[1, ::2] =  1

        covar = matrix.T.dot(matrix)
        inv_cov = scale * np.linalg.inv(covar)
        mu = np.zeros(n)
        mvn = MultivariateNormalFullCovariance(loc=mu, covariance_matrix=inv_cov)

    sample = mvn.sample((batch_size * FLAGS.noise_sim * FLAGS.latent_dim,))
    return sample


def construct_energy_heatmap(target_vars, it, sess):
    # Construct a heat map of transition energies
    energy_pos = target_vars['energy_pos']
    X = target_vars['X']

    base_path = osp.join(FLAGS.logdir, FLAGS.exp)

    n = 300
    lim = 1
    x, y = np.meshgrid(np.linspace(-lim, lim, n), np.linspace(-lim, lim, n))
    coord = np.concatenate([x.flatten()[:, None], y.flatten()[:, None]], axis=1)

    start = coord - 0.01
    end = coord + 0.01

    x_traj = np.stack([start, end], axis=1)[:, :, None, :]

    energy_pos = sess.run([energy_pos], {X: x_traj})[0]
    energy_pos = energy_pos.reshape((n, n))

    lim_labels = np.round(np.linspace(-lim, lim, n), 2)
    df = pd.DataFrame(data=energy_pos, index=lim_labels, columns=lim_labels)
    ax = sns.heatmap(df)
    ax.invert_yaxis()
    plt.savefig(osp.join(base_path, "energy_{}.png".format(it)))
    plt.clf()


def rescale_ob(ob, forward=True):
    # Takes in observation and rescales it be in a good range
    # by either normalizing (forward = True) or unnormalizing (forward = False)
    shape = ob.shape
    ob_flat = ob.reshape((-1, shape[-1]))

    if FLAGS.datasource == "continual_reacher":
        if forward:
            ob_flat[:, 7:14] = ob_flat[:, 7:14] / 10.
        else:
            ob_flat[:, 7:14] = ob_flat[:, 7:14] * 10.

    ob = ob_flat.reshape(shape)

    return ob

def get_avg_step_num(target_vars, sess, env):
    n_exp = FLAGS.n_benchmark_exp
    cond = 'True' if FLAGS.cond else 'False'
    collected_trajs = []

    if FLAGS.linear_inverse_dynamics:
        state_matrix = np.load(osp.join(FLAGS.logdir, FLAGS.exp, 'model_{}.npy'.format(FLAGS.resume_iter)))


    for i in range(n_exp):
        points = []
        cum_rewards = []
        cum_reward = 0
        obs = env.reset()
        start = obs
        ims = []

        action = np.zeros((1, 7))

        while True:
            current_point = obs

            if FLAGS.datasource in ["point", "maze"]:
                end_point = env.end
            else:
                end_point = env.model.site_pos[env.target_sid]

            X_START = target_vars['X_START']
            X_END = target_vars['X_END']
            X_PLAN = target_vars['X_PLAN']
            x_joint = target_vars['x_joint']
            output_actions = target_vars['actions']
            l2_weight = target_vars['l2_weight']
            num_steps = target_vars['num_steps']
            ACTION_PLAN = target_vars['ACTION_PLAN']
            cum_energies = target_vars['cum_plan_energy']

            x_start = current_point[None, None, None, :]
            x_end = end_point[None, None, None, :]

            if FLAGS.smooth_path:
                x_plan = np.tile(x_start, (1, FLAGS.plan_steps, 1, 1))
            else:
                # x_end = np.random.uniform(-1, 1, (FLAGS.num_env, 1, 1, FLAGS.latent_dim))
                # frac = (np.arange(FLAGS.plan_steps) / (FLAGS.plan_steps - 1))[None, :, None, None]
                # x_plan = (1 - frac) * x_start + frac * x_end
                x_plan = np.random.uniform(-1, 1, (1, FLAGS.plan_steps, 1, FLAGS.latent_dim))

            if FLAGS.cond:
                actions = np.random.uniform(-1.0, 1.0, (1, FLAGS.plan_steps + 1, FLAGS.action_dim))
                x_joint, actions = sess.run([x_joint, output_actions],
                                            {X_START: x_start, X_END: x_end,
                                             X_PLAN: x_plan, ACTION_PLAN: actions})
            else:
                ACTION_PLAN = target_vars['ACTION_PLAN']
                actions = np.random.uniform(-0.1, 0.1, (1, FLAGS.plan_steps, FLAGS.action_dim))
                x_joint, output_actions, output_energy = sess.run([x_joint, output_actions, cum_energies],
                        {X_START: x_start, X_END: x_end, X_PLAN: x_plan, l2_weight: 1.0, num_steps: FLAGS.num_plan_steps, ACTION_PLAN: actions})

                print("Output energies ", output_energy)
                # output_actions = output_actions[None, :, :]

            if FLAGS.linear_inverse_dynamics:
                output_actions = linear_reacher_inverse_dynamics(x_joint[:, :, 0, :], action, state_matrix)

            kill = False

            if FLAGS.cond:
                for i in range(actions.shape[1] - 1):
                    obs, reward, done, _ = env.step(actions[0, i, :])
                    target_obs = x_joint[0, i + 1, 0]
                    cum_reward += reward

                    print("obs", obs)
                    print("actions", actions[0, i, :])
                    print("target_obs", target_obs)
                    print("done?", done)
                    points.append(obs)

                    if done:
                        kill = True
                        break

                    if np.abs(target_obs - obs).mean() > 0.55:
                        break

            else:
                for i in range(x_joint.shape[1]-1):
                    if FLAGS.datasource == "continual_reacher" and FLAGS.gt_inverse_dynamics:
                        target = x_joint[:, i+1, 0, :]
                        trajectory = np.concatenate([obs.squeeze()[None, None, :], target[:, None, :]], axis=1)
                        action = process_trajectory(trajectory[0], large=True).squeeze()
                        print("action shape ", action.shape)

                        obs, reward, done, _ = env.step(action)
                    else:
                        action = output_actions[:, i]
                        print(action)
                        obs, reward, done, _ = env.step(action)

                    target_obs = x_joint[0, i + 1, 0]
                    cum_reward += reward

                    print("obs", obs)
                    print("target_obs", target_obs)
                    print("done?", done, "iter {}".format(i))
                    points.append(obs)

                    if done:
                        kill = True
                        break

                    if FLAGS.datasource == "continual_reacher" and FLAGS.render_image:
                        im = env.sim.render(256, 256)
                        ims.append(im)

                    if FLAGS.datasource == "continual_reacher":
                        diff_dist = 0.3
                    else:
                        diff_dist = 0.55

                    if FLAGS.linear_inverse_dynamics and len(points) > 1:
                        update_linear_weight(points[-2][None, :], points[-1][None, :], action[None, :], state_matrix, 0.1)

                    if np.abs(target_obs - obs).mean() > diff_dist:
                        print("replanning")
                        break

            print("done")

            if kill:
                break

            # Only score environments for length equal to 1000
            if FLAGS.score_reward and len(points) > 1000:
                break

        if FLAGS.datasource == "continual_reacher" and FLAGS.render_image:
            print(obs[-6:])
            imageio.mimwrite("traj.gif", ims)
            print("Saved trajectory at traj.gif!!!")
            assert False

        cum_rewards.append(cum_reward)

        # log number of steps for each experiment
        ts = str(datetime.datetime.now())
        d = {'ts': ts,
             'start': start,
             'actual_end': x_start,
             'end': x_end,
             'obstacle': FLAGS.obstacle,
             'cond': cond,
             'plan_steps': FLAGS.plan_steps,
             'step_num': len(points),
             'exp': FLAGS.exp,
             'iter': FLAGS.resume_iter}
        log_step_num_exp(d)

        collected_trajs.append(np.array(points))

    imgdir = FLAGS.imgdir
    if not osp.exists(imgdir):
        os.makedirs(imgdir)

    lengths = []

    if FLAGS.score_reward:
        print("Obtained an average reward of {} for {} runs on enviroment {}".format(np.mean(cum_rewards),
                                                                                     FLAGS.n_benchmark_exp,
                                                                                     FLAGS.datasource))

    if FLAGS.log_traj:
        for traj in collected_trajs:
            traj = traj.squeeze()
            if traj.ndim == 1:
                traj = np.expand_dims(traj, 0)

            # save one image for each trajectory
            timestamp = str(datetime.datetime.now())

            if FLAGS.obstacle != None:
                xy = (FLAGS.obstacle[0], FLAGS.obstacle[-1])
                w, h = FLAGS.obstacle[2] - FLAGS.obstacle[0], FLAGS.obstacle[1] - FLAGS.obstacle[3]

                # create a Rectangle patch as obstacle
                if FLAGS.datasource == "point":
                    ax = plt.gca()  # get the current reference
                    rect = patches.Rectangle(xy, w, h, linewidth=1, edgecolor='r', facecolor='none')
                    ax.add_patch(rect)
                elif FLAGS.datasource == "maze":
                    # Plot the values of boundaries of the maze
                    ax = plt.gca()
                    rect = patches.Rectangle((-0.75, -1.0), 0.25, 1.75, linewidth=1, edgecolor='r', facecolor='none')
                    ax.add_patch(rect)
                    rect = patches.Rectangle((-0.25, -0.75), 0.25, 1.75, linewidth=1, edgecolor='r', facecolor='none')
                    ax.add_patch(rect)
                    rect = patches.Rectangle((0.25, -1.0), 0.25, 1.75, linewidth=1, edgecolor='r', facecolor='none')
                    ax.add_patch(rect)
                    rect = patches.Rectangle((0.75, -0.75), 0.25, 1.75, linewidth=1, edgecolor='r', facecolor='none')
                    ax.add_patch(rect)

            plt.plot(traj[:, 0], traj[:, 1], color='green', alpha=0.3)

            if FLAGS.save_single:
                save_dir = osp.join(imgdir, 'test_{}_iter{}_{}.png'.format(FLAGS.exp, FLAGS.resume_iter, timestamp))
                plt.savefig(save_dir)
                plt.clf()

            # save all length for calculation of average length
            lengths.append(traj.shape[0])

        if not FLAGS.save_single:
            save_dir = osp.join(imgdir, 'benchmark_{}_{}_iter{}_{}'.format(FLAGS.n_benchmark_exp, FLAGS.exp,
                                                                           FLAGS.resume_iter, timestamp))
            if FLAGS.constraint_vel:
                save_dir += "_vel"
            if FLAGS.constraint_goal:
                save_dir += "_goal"
            save_dir += ".png"

            plt.savefig(save_dir)
            plt.clf()

        average_length = sum(lengths) / len(lengths)
        print("average number of steps:", average_length)


def rl_train(target_vars, saver, sess, logger, resume_iter, env):
    tot_iter = int(FLAGS.nsteps // FLAGS.num_env)

    X = target_vars['X']
    X_NOISE = target_vars['X_NOISE']
    train_op = target_vars['train_op']
    loss_ml = target_vars['loss_ml']
    x_grad = target_vars['x_grad']
    x_mod = target_vars['x_mod']
    action_grad = target_vars['action_grad']
    X_START = target_vars['X_START']
    X_END = target_vars['X_END']
    X_PLAN = target_vars['X_PLAN']
    ACTION_PLAN = target_vars['ACTION_PLAN']
    ACTION_LABEL = target_vars['ACTION_LABEL']
    ACTION_NOISE = target_vars['ACTION_NOISE_LABEL']
    x_joint = target_vars['x_joint']
    actions = target_vars['actions']
    energy_pos = target_vars['energy_pos']
    energy_neg = target_vars['energy_neg']
    loss_total = target_vars['total_loss']
    dyn_loss = target_vars['dyn_loss']
    dyn_dist = target_vars['dyn_dist']

    if not FLAGS.ff_model:
        cum_energies = target_vars['cum_plan_energy']
    else:
        cum_energies = tf.zeros(1)

    num_steps = target_vars['num_steps']
    LR = target_vars['lr']
    l2_weight = target_vars['l2_weight']
    num_steps = target_vars['num_steps']

    num_plan_steps = FLAGS.num_plan_steps

    ob = env.reset()[:, None, None, :]

    output = [train_op, x_mod]
    log_output = [train_op, dyn_loss, dyn_dist, energy_pos, energy_neg, loss_ml, loss_total, x_grad, action_grad, x_mod]

    replay_buffer = ReplayBuffer(10000)
    pos_replay_buffer = ReplayBuffer(1000000)

    epinfos = []
    points = []
    total_obs = []

    if FLAGS.smooth_path:
        x_traj = np.tile(ob, (1, FLAGS.plan_steps+1, 1, 1))
    else:
        x_traj = np.random.uniform(-1.0, 1.0, (FLAGS.num_env, FLAGS.plan_steps+1, 1, FLAGS.latent_dim))

    action_plan = np.random.uniform(-1, 1, (FLAGS.num_env, FLAGS.plan_steps, FLAGS.action_dim))
    plan_energy = np.zeros(FLAGS.num_env)

    state_matrix = np.zeros((17, 24))
    env_action = None
    num_env_steps = 0
    heatmap_counter = 0

    coord = np.linspace(-1, 1, 10)
    x, y, z = np.meshgrid(coord, coord, coord)
    cube_3d_low = np.stack([x, y, z], axis=3)
    cube_3d_high = np.stack([x, y, z], axis=3) + 4 / 20

    cube_3d_low = cube_3d_low.reshape((-1, 3))
    cube_3d_high = cube_3d_high.reshape((-1, 3))

    occupancy = np.zeros([cube_3d_low.shape[0], 1], dtype=np.bool)

    prev_bp = 0
    cum_bp = 0
    for itr in range(resume_iter, tot_iter):
        if itr != resume_iter:
            if FLAGS.smooth_path:
                x_traj_random = np.tile(ob, (1, FLAGS.plan_steps+1, 1, 1))
            else:
                x_traj_random = np.random.uniform(-1.0, 1.0, (FLAGS.num_env, FLAGS.plan_steps+1, 1, FLAGS.latent_dim))

            # print(plan_energy.squeeze() / FLAGS.num_plan_steps)
            if not FLAGS.ff_model:
                dones_tot = dones_tot | (plan_energy.sum(axis=1) / FLAGS.num_plan_steps > 1.0)
                x_traj[dones_tot] = x_traj_random[dones_tot]

        if FLAGS.ff_model:
            x_plan = np.random.uniform(-1.0, 1.0, (FLAGS.num_env, FLAGS.plan_steps, 1, FLAGS.latent_dim))
        else:
            # print("x_traj, x_plan shape ", x_traj.shape, x_plan.shape)
            print("x_traj shape ", x_traj.shape, prev_bp)
            print("shapes ", x_traj[:, 1+prev_bp+1:].shape, np.tile(x_traj[:, -1:], (1, prev_bp+1, 1, 1)).shape)
            x_plan = np.concatenate([x_traj[:, 1+prev_bp+1:], np.tile(x_traj[:, -1:], (1, prev_bp+1, 1, 1))], axis=1)
        action_plan = np.concatenate([action_plan[:, 1:], action_plan[:, -1:]], axis=1)

        if FLAGS.datasource == "maze":
            x_end = np.tile(np.array([[0.7, -0.8]]), (FLAGS.num_env, 1))[:, None, None, :]
        elif FLAGS.datasource == "reacher":
            x_end = np.tile(np.array([[0.7, 0.5]]), (FLAGS.num_env, 1))[:, None, None, :]
        elif FLAGS.datasource == "continual_reacher":
            x_end = ob[:, :, :, -6:-3]
        else:
            x_end = np.tile(np.array([[0.5, 0.5]]), (FLAGS.num_env, 1))[:, None, None, :]

        feed_dict = {X_START: ob, X_PLAN: x_plan, X_END: x_end, ACTION_PLAN: action_plan}
        feed_dict[l2_weight] = min(itr / 200, 1)
        feed_dict[num_steps] = num_plan_steps

        x_traj, traj_actions, plan_energy = sess.run([x_joint, actions, cum_energies], feed_dict)

        # Clip actions in the case of continual reacher
        # if FLAGS.datasource == "continual_reacher" or FLAGS.datasource == "reacher":
        #     traj_actions = traj_actions + np.random.uniform(-0.05, 0.05, traj_actions.shape)
        #     traj_actions = np.clip(traj_actions, -1, 1)

        if FLAGS.debug:
            print(x_traj[0])

        if FLAGS.datasource == "continual_reacher" and FLAGS.linear_inverse_dynamics:
            if env_action is not None:
                traj_actions = linear_reacher_inverse_dynamics(x_traj.squeeze(), env_action, state_matrix)
                # print(traj_actions.shape)
                # assert False
                traj_actions = np.clip(traj_actions, -1, 1)
                # print(state_matrix)
        elif FLAGS.datasource == "continual_reacher" and FLAGS.gt_inverse_dynamics:
            traj_actions = continual_reacher_inverse_dynamics(x_traj.squeeze())

        obs = [ob[:, 0, 0, :]]
        dones = []
        diffs = []
        dones_tot = np.zeros(FLAGS.num_env).astype(np.bool)

        if FLAGS.adaptive_sample:
            print("Plan energy ", plan_energy.mean(axis=0))

        for bp in range(traj_actions.shape[1]):
            num_env_steps += 1

            if FLAGS.random_action:
                action = np.random.uniform(-1, 1, traj_actions[:, bp].shape)
            else:
                action = traj_actions[:, bp]

            env_action = action
            ob, _, done, infos = env.step(action)

            dones_tot = (done | dones_tot)

            # print("Hand position ", ob[0])
            # print("Action ", action[0])
            if bp == 0 and FLAGS.datasource != "continual_reacher":
                # print(x_traj[0, 0], x_traj[0, 1], ob[0])
            #     target_ob = x_traj[:, bp + 1, 0]
                print("x_traj", x_traj[0, :, 0])
            #     print("Abs dist: ", np.mean(np.abs(ob - target_ob)))
            # #     print("Trajectory: ", x_traj[0, -2:])

            dones.append(done)
            obs.append(ob)

            for info in infos:
                maybeepinfo = info.get('episode')
                if maybeepinfo: epinfos.append(maybeepinfo)

            # print(ob.shape)
            if FLAGS.datasource == "continual_reacher":
                diff = np.abs(x_traj[:, bp + 1, 0, :-3] - ob[:, :-3]).mean()
            else:
                diff = np.abs(x_traj[:, bp + 1, 0] - ob).max()

            diffs.append(diff)

            if FLAGS.datasource == "continual_reacher":
                diff_env = 2000
            else:
                diff_env = 0.7

            if FLAGS.end_effector_stat and FLAGS.datasource == "continual_reacher":
                ob_effector = ob[:, -6:-3]
                low_mask = (ob_effector[:, None, :] > cube_3d_low[None, :, :]).all(axis=2)
                high_mask = (ob_effector[:, None, :] < cube_3d_high[None, :, :]).all(axis=2)
                occupancy = occupancy | (low_mask & high_mask).any(axis=0)[:, None]

            if (((diff > diff_env)) or dones_tot.any()) and not FLAGS.random_action:
                # print("Broke on ")
                if prev_bp == bp:
                    cum_bp += 1
                else:
                    cum_bp = 0
                    prev_bp = bp
                break

            if FLAGS.linear_inverse_dynamics:
                state_matrix = update_linear_weight(obs[-2], obs[-1], action, state_matrix, 0.1)

        ob = ob[:, None, None, :]
        dones = np.array(dones).transpose()
        obs = np.stack(obs, axis=1)[:, :, None, :]

        if FLAGS.heatmap:
            total_obs.append(obs.reshape((-1, FLAGS.latent_dim)))

        if FLAGS.datasource == "point" or FLAGS.datasource == "maze":
            # Rescale the action so there isn't ambiguity
            traj_actions_max = np.abs(traj_actions).max(axis=2, keepdims=True)
            traj_actions = traj_actions / np.maximum(traj_actions_max, 1)

        action, ob_pair = parse_valid_obs(obs, traj_actions, dones)
        # print("Maximum and minimum of observation: ", np.amax(ob_pair, axis=(0, 1, 2)).round(decimals=1), np.amin(ob_pair, axis=(0, 1, 2)).round(decimals=1))

        # np.save("debug.npy", ob_pair)

        # For now only encode the first negative transition

        if np.random.uniform() < -0.5:
            x_noise = np.stack([x_traj[:, :-1], x_traj[:, 1:]], axis=2)
            action_noise_neg = traj_actions[:, :]
        else:
            n = min(bp+1, x_traj.shape[1]-2)
            x_noise = np.stack([x_traj[:, :n], x_traj[:, 1:n+1]], axis=2)
            action_noise_neg = traj_actions[:, :n]

        # whole trajectory instead
        # x_noise = np.stack([x_traj[:, :-1], x_traj[:, 1:]], axis=2)
        # action_noise_neg = traj_actions[:, :]

        s = x_noise.shape
        x_noise_neg = x_noise.reshape((s[0] * s[1], s[2], s[3], s[4]))
        s = action_noise_neg.shape
        action_noise_neg = action_noise_neg.reshape((s[0] * s[1], s[2]))


        if ob_pair is not None:
            pos_batch = ob_pair.shape[0]
        else:
            pos_batch = FLAGS.num_env * FLAGS.batch_size

        if ob_pair is not None:
            traj_action_encode = action.reshape((-1, 1, 1, FLAGS.action_dim))
            encode_data = np.concatenate([ob_pair, np.tile(traj_action_encode, (1, FLAGS.total_frame, 1, 1))], axis=3)
            pos_replay_buffer.add(encode_data)

        if len(pos_replay_buffer) > pos_batch:
            sample_data = pos_replay_buffer.sample(pos_batch)
            sample_ob = sample_data[:, :, :, :-FLAGS.action_dim]
            sample_actions = sample_data[:, 0, 0, -FLAGS.action_dim:]

            if ob_pair is not None:
                # replay_mask = (np.random.uniform(0, 1, (pos_batch)) > 0.5)
                ob_pair = np.concatenate([ob_pair, sample_ob], axis=0)
                action = np.concatenate([action, sample_actions], axis=0)
                # ob_pair[replay_mask] = sample_ob[replay_mask]
                # action[replay_mask] = sample_actions[replay_mask]
            else:
                ob_pair, action = sample_ob, sample_actions

        if FLAGS.record_reacher_data:
            print("Length of the pos_replay_buffer is ", len(pos_replay_buffer))
            if FLAGS.datasource == "continual_reacher" and len(pos_replay_buffer) > 120000:
                dat = np.array(pos_replay_buffer._storage[:120000])
                sample_ob = dat[:, :, 0, :-FLAGS.action_dim]
                sample_actions = dat[:, 0, 0, -FLAGS.action_dim:]
                np.savez("data/continual_reacher_model.npz", ob=sample_ob, action=sample_actions)
                assert False

        feed_dict = {X: ob_pair, X_NOISE: x_noise_neg, ACTION_NOISE: action_noise_neg, LR: FLAGS.lr}
        # print("x_noise_neg ", x_noise_neg)

        if ACTION_LABEL is not None:
            feed_dict[ACTION_LABEL] = action

        if ob_pair is None:
            continue

        batch_size = x_noise_neg.shape[0]
        if FLAGS.replay_batch and len(replay_buffer) > batch_size and not FLAGS.ff_model:
            replay_batch = replay_buffer.sample(int(batch_size))
            replay_mask = (np.random.uniform(0, 1, (batch_size)) > 0.5)
            # feed_dict[X_NOISE][replay_mask] = replay_batch[replay_mask]
            feed_dict[X_NOISE][replay_mask] = replay_batch[replay_mask]

        # print("Normalization statistics ", feed_dict[X_NOISE].mean(axis=(0, 1)), feed_dict[X_NOISE].std(axis=(0, 1)))

        if num_env_steps > 200 * heatmap_counter and FLAGS.energy_heatmap:
            construct_energy_heatmap(target_vars, num_env_steps, sess)
            total_obs_concat = np.concatenate(total_obs, axis=0)
            ax = plt.gca()
            # ax.set_ylim(-1, 1)
            # ax = sns.kdeplot(data=total_obs_concat[:, 0], data2=total_obs_concat[:, 1], shade=True, ax=ax)

            n = 20
            lim = 1
            x, y = np.meshgrid(np.linspace(-lim, lim, n), np.linspace(-lim, lim, n))
            coord_low = np.concatenate([x.flatten()[:, None], y.flatten()[:, None]], axis=1)
            coord_high = coord_low + (2*lim) / n
            low_mask = (total_obs_concat[:, None, :] > coord_low[None, :, :]).all(axis=2)
            high_mask = (total_obs_concat[:, None, :] < coord_high[None, :, :]).all(axis=2)
            joint_mask = (low_mask & high_mask).astype(np.int32)
            # print("joint_mask sum ", joint_mask.sum())

            # if itr >= 100:
            #     assert False

            counts = joint_mask.sum(axis=0).reshape((n, n))
            counts = np.log(counts+1)

            lim_labels = np.round(np.linspace(-lim, lim, n), 2)
            df = pd.DataFrame(data=counts, index=lim_labels, columns=lim_labels)
            ax = sns.heatmap(df)
            ax.invert_yaxis()

            plt.savefig(osp.join(FLAGS.logdir, FLAGS.exp, "kde_{}.png".format(num_env_steps)))
            plt.clf()

            heatmap_counter += 1

        if itr % FLAGS.log_interval == 0:
            _, dyn_loss, dyn_dist, e_pos, e_neg, loss_ml, loss_total, x_grad, action_grad, x_mod = sess.run(log_output,
                                                                                                            feed_dict=feed_dict)

            if (e_neg.mean() - e_pos.mean()) > 0.15 and FLAGS.adaptive_sample:
                num_plan_steps += 20
                num_plan_steps = min(num_plan_steps, 800)

            kvs = {}
            kvs['e_pos'] = e_pos.mean()
            kvs['e_neg'] = e_neg.mean()
            kvs['loss_ml'] = loss_ml.mean()
            kvs['loss_total'] = loss_total.mean()
            kvs['x_grad'] = np.abs(x_grad).mean()
            kvs['action_grad'] = np.abs(action_grad).mean()
            kvs['dyn_loss'] = dyn_loss.mean()
            kvs['dyn_dist'] = np.abs(dyn_dist).mean()
            kvs['iter'] = itr
            kvs["train_episode_length_mean"] = safemean([epinfo['l'] for epinfo in epinfos])
            kvs["train_episode_reward_mean"] = safemean([epinfo['r'] for epinfo in epinfos])
            kvs["diffs_start"] = diffs[0]
            kvs["diffs_end"] = diffs[-1]
            kvs["num_plan_steps"] = num_plan_steps
            kvs["num_env_steps"] = num_env_steps
            kvs["end_effector_occupancy"] = occupancy.sum()

            epinfos = []

            string = "Obtained a total of "
            for key, value in kvs.items():
                string += "{}: {}, ".format(key, value)

            print(string)
            logger.writekvs(kvs)
        else:
            _, x_mod = sess.run(output, feed_dict=feed_dict)

        # print("x_mod ", x_mod)

        if FLAGS.replay_batch:
            replay_buffer.add(x_mod)

        if itr % FLAGS.save_interval == 0:
            saver.save(sess, osp.join(FLAGS.logdir, FLAGS.exp, 'model_{}'.format(itr)))

            if FLAGS.linear_inverse_dynamics:
                np.save(osp.join(FLAGS.logdir, FLAGS.exp, 'model_{}.npy'.format(itr)), state_matrix)

        if FLAGS.heatmap and num_env_steps > 4000:
            total_obs = np.concatenate(total_obs, axis=0)

            n = 20
            lim = 1
            x, y = np.meshgrid(np.linspace(-lim, lim, n), np.linspace(-lim, lim, n))
            coord_low = np.concatenate([x.flatten()[:, None], y.flatten()[:, None]], axis=1)
            coord_high = coord_low + (2*lim) / n
            low_mask = (total_obs_concat[:, None, :] > coord_low[None, :, :]).all(axis=2)
            high_mask = (total_obs_concat[:, None, :] < coord_high[None, :, :]).all(axis=2)
            joint_mask = (low_mask & high_mask).astype(np.int32)
            # print("joint_mask sum ", joint_mask.sum())

            # if itr >= 100:
            #     assert False

            counts = joint_mask.sum(axis=0).reshape((n, n))
            counts = np.log(counts+1)

            lim_labels = np.round(np.linspace(-lim, lim, n), 2)
            df = pd.DataFrame(data=counts, index=lim_labels, columns=lim_labels)
            ax = sns.heatmap(df)
            ax.invert_yaxis()

            plt.savefig(osp.join(FLAGS.logdir, FLAGS.exp, "kde_{}.png".format(num_env_steps)))
            plt.clf()
            assert False


def train(target_vars, saver, sess, logger, dataloader, actions, resume_iter, mask=None):
    X = target_vars['X']
    X_NOISE = target_vars['X_NOISE']
    train_op = target_vars['train_op']
    energy_pos = target_vars['energy_pos']
    energy_neg = target_vars['energy_neg']
    loss_energy = target_vars['loss_energy']
    loss_ml = target_vars['loss_ml']
    loss_total = target_vars['total_loss']
    gvs = target_vars['gvs']
    x_grad = target_vars['x_grad']
    action_grad = target_vars['action_grad']
    x_off = target_vars['x_off']
    temp = target_vars['temp']
    x_mod = target_vars['x_mod']
    weights = target_vars['weights']
    lr = target_vars['lr']
    ACTION_LABEL = target_vars['ACTION_LABEL']
    ACTION_NOISE_LABEL = target_vars['ACTION_NOISE_LABEL']
    dyn_loss = target_vars['dyn_loss']
    dyn_dist = target_vars['dyn_dist']
    progress_diff = target_vars['progress_diff']
    ff_loss = target_vars['ff_loss']
    ff_dist = target_vars['ff_dist']
    MASK = target_vars['mask']

    n = FLAGS.n_exp
    gvs_dict = dict(gvs)

    # remove gradient logging since it is slow
    log_output = [train_op, dyn_loss, dyn_dist, energy_pos, energy_neg, loss_energy, loss_ml, loss_total, x_grad,
                  action_grad, x_off, x_mod, progress_diff, ff_loss, ff_dist,
                  *gvs_dict.keys()]
    output = [train_op, x_mod]

    itr = resume_iter
    x_mod = None
    gd_steps = 1

    print(dataloader.shape)

    # Generated correlated experience by tiling values a bunch of times
    if not FLAGS.random_shuffle:
        dataloader = np.tile(dataloader[:, None, :, :, :], (1, 100, 1, 1, 1))
        actions = np.tile(actions[:, None, :, :], (1, 100, 1, 1))

        dataloader = dataloader.reshape((-1, *dataloader.shape[2:]))
        actions = actions.reshape((-1, *actions.shape[2:]))

    random_combo = list(product(range(FLAGS.total_frame, dataloader.shape[1]+1),
                                range(0, dataloader.shape[0] - FLAGS.batch_size, FLAGS.batch_size)))

    replay_buffer = ReplayBuffer(10000)

    for epoch in range(FLAGS.epoch_num):
        if FLAGS.random_shuffle:
            random.shuffle(random_combo)
        # perm_idx = np.random.permutation(dataloader.shape[0])
        perm_idx = np.arange(dataloader.shape[0], dtype=np.int32)

        for j, i in random_combo:
            label = dataloader[:, j - FLAGS.total_frame:j]
            label_i = label[perm_idx[i:i + FLAGS.batch_size]]
            # data_corrupt = np.random.uniform(-1.0, 1.0, (
            #     FLAGS.batch_size, FLAGS.total_frame, FLAGS.input_objects, FLAGS.latent_dim))

            data_corrupt = np.random.uniform(-1.0, 1.0, (
                FLAGS.batch_size, 1, FLAGS.input_objects, FLAGS.latent_dim))

            data_corrupt = np.tile(data_corrupt, (1, FLAGS.total_frame, 1, 1))

            feed_dict = {X: label_i, X_NOISE: data_corrupt, lr: FLAGS.lr}

            if mask is not None:
                mask_val = mask[perm_idx[i:i + FLAGS.batch_size], j - FLAGS.total_frame:j]
                feed_dict[MASK] = mask_val
                # print("mask shape ", mask_val.shape)

            if ACTION_LABEL is not None:
                feed_dict[ACTION_LABEL] = actions[perm_idx[i:i + FLAGS.batch_size], j - FLAGS.total_frame + 1]
                feed_dict[ACTION_NOISE_LABEL] = np.random.uniform(-1.2, 1.2, (FLAGS.batch_size, FLAGS.action_dim))

            # print("Action label", feed_dict[ACTION_LABEL][0])
            # print("Action noise label", feed_dict[ACTION_NOISE_LABEL][0])
            # print(label_i.shape)
            # print("X label", (label_i[:, 1, 0] - label_i[:, 0, 0]) - feed_dict[ACTION_LABEL] / 20)
            # assert False

            if len(replay_buffer) > FLAGS.batch_size and FLAGS.replay_batch and not FLAGS.ff_model:
                replay_batch = replay_buffer.sample(FLAGS.batch_size)
                replay_mask = (np.random.uniform(0, 1, (FLAGS.batch_size)) > 0.05)
                # replay_batch = np.clip(replay_batch, -1, 1)
                data_corrupt[replay_mask] = replay_batch[replay_mask]

            if itr % FLAGS.log_interval == 0:
                _, dyn_loss, dyn_dist, e_pos, e_neg, loss_e, loss_ml, loss_total, x_grad, action_grad, x_off, x_mod, progress_diff, ff_loss, ff_dist, *grads = sess.run(
                    log_output, feed_dict)

                kvs = {}
                kvs['e_pos'] = e_pos.mean()
                kvs['temp'] = temp
                kvs['e_neg'] = e_neg.mean()
                kvs['loss_e'] = loss_e.mean()
                kvs['dyn_loss'] = dyn_loss.mean()
                kvs['dyn_dist'] = dyn_dist.mean()
                kvs['progress_diff'] = progress_diff.mean()

                kvs['loss_ml'] = loss_ml.mean()
                kvs['loss_total'] = loss_total.mean()
                kvs['x_grad'] = np.abs(x_grad).mean()
                kvs['action_grad'] = np.abs(action_grad).mean()
                kvs['x_off'] = x_off.mean()
                kvs['iter'] = itr
                kvs['ff_loss'] = np.abs(ff_loss).mean()
                kvs['ff_dist'] = np.abs(ff_dist).mean()

                for v, k in zip(grads, [v.name for v in gvs_dict.values()]):
                    kvs[k] = np.abs(v).mean()

                string = "Obtained a total of "
                for key, value in kvs.items():
                    string += "{}: {}, ".format(key, value)

                print(string)

                logger.writekvs(kvs)
            else:
                _, x_mod, *mods = sess.run(output, feed_dict)
                # print([mod[0] for mod in mods])

            if FLAGS.replay_batch and (x_mod is not None):
                replay_buffer.add(x_mod)

            if itr % FLAGS.save_interval == 0:
                saver.save(sess, osp.join(FLAGS.logdir, FLAGS.exp, 'model_{}'.format(itr)))
            itr += 1

    saver.save(sess, osp.join(FLAGS.logdir, FLAGS.exp, 'model_{}'.format(itr)))


def eval_collision(target_vars, sess, dataset_test, mask_test, dataset_test_gt):
    X_PLAN = target_vars['X_PLAN']
    MASK_PLAN = target_vars['mask_plan']
    x_joint = target_vars['x_joint']
    cum_energies = target_vars['cum_energies']

    # Sample a series of random trajectories of a certain length data_test
    errors = []
    baseline_errors = []

    for i in tqdm(range(10)):
        batch = random.randint(0, dataset_test.shape[0] - FLAGS.batch_size)
        it = random.randint(0, dataset_test.shape[1] - FLAGS.plan_steps)

        masked_traj = dataset_test[batch:batch+FLAGS.batch_size, it:it+FLAGS.plan_steps]
        mask = mask_test[batch:batch+FLAGS.batch_size, it:it+FLAGS.plan_steps]

        # print("Mask shape ", mask.shape)
        # print(mask_test.shape)
        # print(dataset_test.shape)
        # assert False

        pred_unmask, plan_energies = sess.run([x_joint, cum_energies], {X_PLAN: masked_traj, MASK_PLAN: mask})
        true_unmask = dataset_test_gt[batch:batch+FLAGS.batch_size, it:it+FLAGS.plan_steps]

        # print(pred_unmask[0, 0])
        # print(true_unmask[0, 0])
        # print(mask[0, 0])
        # assert False

        # print("plan_energies ", plan_energies)

        errors.append(np.abs(pred_unmask.squeeze() - true_unmask).mean())
        baseline_errors.append(np.abs(masked_traj - true_unmask).mean())


    print("Obtained an error of ", np.mean(errors))
    print("The original trajectory has  an error of ", np.mean(baseline_errors))


def test(target_vars, saver, sess, logdir, data, actions, dataset_train, mean, std):
    X_START = target_vars['X_START']
    X_END = target_vars['X_END']
    X_PLAN = target_vars['X_PLAN']
    x_joint = target_vars['x_joint']

    n = FLAGS.n_exp

    if FLAGS.datasource == "point" or FLAGS.datasource == 'phy_a' or FLAGS.datasource == 'phy_cor':
        x_start0, x_start1 = 0.0, 0.0
        x_end0, x_end1 = 1.0, 1.0
        x_start = np.array([x_start0, x_start1])[None, None, None, :]
        x_end = np.array([x_end0, x_end1])[None, None, None, :]
    elif FLAGS.datasource == "maze":
        x_start = np.array([-0.85, -0.85])[None, None, None, :]
        x_end = np.array([0.7, -0.8])[None, None, None, :]
    elif FLAGS.datasource == "reacher":
        # x_start = (np.array([0.00895044, -0.02340578, 0.0, 0.0])[None, None, None, :] - mean) / std
        # x_start = (np.array([0.08984227,  0.06335336, 0.0, 0.0])[None, None, None, :] - mean) / std
        n = 25
        x_start = data[n:n + 1, 3:4]
        # x_end = np.array([0.7, -0.8])[None, None, None, :]
        x_end = data[n:n + 1, 3 + FLAGS.plan_steps + 1:3 + FLAGS.plan_steps + 2]
    elif FLAGS.datasource == "continual_reacher":
        x_start = np.array([3.53e-4, -1.59e-3, 6.077e-2, 3.099e-3, -1.079e-2, 3.217e-2, -3.403e-3, 2.20e-3,
                            2.3671e-3, 3.812e-1, -1.282e-2, -5.663e-2, 8.615e-2, -1.7422e-2, 8.209e-1, -5.998e-1,
                            -8.838e-5, 2.4255e-2, -1.5922e-1, -2.2314e-2])
        x_end = np.array([0.1, 0.1, 0.1])[None, None, None, :]


    x_plan = np.random.uniform(-1, 1, (FLAGS.n_exp, FLAGS.plan_steps, 1, FLAGS.latent_dim))
    x_start, x_end, x_plan = np.tile(x_start, (n, 1, 1, 1)), np.tile(x_end, (n, 1, 1, 1)), x_plan

    if FLAGS.cond:
        ACTION_PLAN = target_vars['ACTION_PLAN']
        actions_tensor = target_vars['actions']
        actions = np.random.uniform(-1.0, 1.0, (n, FLAGS.plan_steps + 1, FLAGS.action_dim))
        x_joint, actions = sess.run([x_joint, actions_tensor],
                                    {X_START: x_start, X_END: x_end, X_PLAN: x_plan, ACTION_PLAN: actions})
    else:
        x_joint = sess.run([x_joint], {X_START: x_start, X_END: x_end, X_PLAN: x_plan})[0]

    print("actions:", actions[0])
    print("x_joint:", x_joint[0])

    imgdir = FLAGS.imgdir
    if not osp.exists(imgdir):
        os.makedirs(imgdir)
    timestamp = str(datetime.datetime.now())

    # Generate 2D plots of particle movement in the environment
    if FLAGS.datasource == "point" or FLAGS.datasource == "maze":
        xs, ys = [], []
        for i in range(n):
            x_joint_i = x_joint[i]
            x, y = zip(*list(x_joint_i.squeeze()))
            xs.append(x)
            ys.append(y)
            plt.plot(x, y, color='blue', alpha=0.1)
        # sns.jointplot(x=xs, y=ys)

        if FLAGS.datasource == "maze":
            ax = plt.gca()
            rect = patches.Rectangle((-0.75, -1.0), 0.25, 1.75, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            rect = patches.Rectangle((-0.25, -0.75), 0.25, 1.75, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            rect = patches.Rectangle((0.25, -1.0), 0.25, 1.75, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            rect = patches.Rectangle((0.75, -0.75), 0.25, 1.75, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)

        save_dir = osp.join(imgdir, 'test_exp_{}_iter{}_{}.png'.format(FLAGS.exp, FLAGS.resume_iter, timestamp))
        # plt.tight_layout()
        plt.xlim(x_start0 - 0.5, x_end0 + 0.2)
        plt.ylim(x_start1 - 0.5, x_end1 + 0.2)
        plt.title("Plan steps = {}".format(FLAGS.plan_steps))
        plt.savefig(save_dir)

    elif FLAGS.datasource == "reacher":
        # Generate 2D Videos of Particles Moving
        dats = x_joint[0]
        dats = dats * std + mean
        env = gym.make("Reacher-v2")
        sim = env.sim
        ims = []

        for i in range(dats.shape[0]):
            state = dats[i]
            sim.data.qpos[:2] = state[0, :2]
            sim.forward()
            im = sim.render(256, 256)
            ims.append(im)

        save_file = osp.join(imgdir, 'test_exp{}_iter{}_{}.gif'.format(FLAGS.exp, FLAGS.resume_iter, timestamp))
        imageio.mimwrite(save_file, ims)

    elif FLAGS.datasource == 'phy_a':
        xs, ys, ps = [], [], []
        for i in range(n):
            x_joint_i = x_joint[i]
            x, y, p = zip(*list(x_joint_i.squeeze()))
            xs.append(x)
            ys.append(y)
            ps.append(p)
            print("phy param", p)

            plt.plot(x, y, color='blue', alpha=0.1)

        save_dir = osp.join(imgdir, 'test_phy_a.png'.format(FLAGS.exp, FLAGS.resume_iter, timestamp))
        # plt.tight_layout()
        plt.xlim(x_start0 - 0.5, x_end0 + 0.2)
        plt.ylim(x_start1 - 0.5, x_end1 + 0.2)
        plt.title("Plan steps = {}".format(FLAGS.plan_steps))
        plt.savefig(save_dir)


    elif FLAGS.datasource == "continual_reacher":
        # Generate 2D Videos of Particles Moving
        dats = x_joint[0]
        env = ContinualReacher7DOFEnv()
        sim = env.sim
        # env.sim.data.site_xpos[env.target_sid] = [0.1, 0.1, 0.1]
        ims = []

        for i in range(dats.shape[0]):
            state = dats[i]
            sim.data.qpos[:7] = state[0, :7]
            sim.forward()
            im = sim.render(256, 256)
            ims.append(im)

        save_file = osp.join(imgdir, 'test_exp{}_iter{}_{}.gif'.format(FLAGS.exp, FLAGS.resume_iter, timestamp))
        imageio.mimwrite(save_file, ims)


def debug(target_vars, sess):
    X = target_vars['X']
    ACTION_LABEL = target_vars['ACTION_LABEL']
    energy_pos = target_vars['energy_pos']
    n = 100

    debug_frame = np.array([[0.05, 0.05], [0.0, 0.0]])[None, :, None, :]
    actions_sample = np.linspace(-1, 1, n)
    actions_x, actions_y = np.meshgrid(actions_sample, actions_sample)
    actions_tot = np.stack([actions_x[:, :, None], actions_y[:, :, None]], axis=2)

    actions_label = actions_tot.reshape((-1, 2))
    debug_frame = np.tile(debug_frame, (actions_label.shape[0], 1, 1, 1))
    energy_pos = sess.run([energy_pos], {X: debug_frame, ACTION_LABEL: actions_label})[0]
    energy_pos = energy_pos.reshape((n, n))
    plt.imshow(energy_pos, cmap='hot', interpolation='nearest')
    plt.savefig("cmap.png")


def construct_plan_model(model, weights, X_PLAN, X_START, X_END, ACTION_LABEL, cond=False, target_vars={}):
    x_joint = tf.concat([X_START, X_PLAN], axis=1)
    steps = tf.constant(0)

    l2_weight = tf.placeholder(shape=None, dtype=tf.float32)
    num_steps = tf.placeholder(shape=None, dtype=tf.float32)
    target_vars['l2_weight'] = l2_weight
    target_vars['num_steps'] = num_steps

    c = lambda i, x, y: tf.less(tf.to_float(i), num_steps)

    if cond:
        actions = ACTION_LABEL
    else:
        actions = tf.zeros(1)

    def mcmc_step(counter, x_joint, actions):
        if cond:
            actions = actions + tf.random_normal(tf.shape(actions), mean=0.0, stddev=0.01)

        x_joint = x_joint + tf.random_normal(tf.shape(x_joint), mean=0.0, stddev=0.01)
        cum_energies = 0
        for i in range(FLAGS.plan_steps - FLAGS.total_frame + 2):
            if cond:
                cum_energy = model.forward(x_joint[:, i:i + FLAGS.total_frame], weights, action_label=actions[:, i], opt_low=FLAGS.opt_low)
            else:
                cum_energy = (FLAGS.plan_steps - i) * model.forward(x_joint[:, i:i + FLAGS.total_frame], weights, action_label=None, opt_low=FLAGS.opt_low)

            cum_energies = cum_energies + cum_energy

        if FLAGS.anneal:
            anneal_val = tf.cast(counter, tf.float32) / FLAGS.num_plan_steps
        else:
            anneal_val = 1

        if FLAGS.constraint_vel:
            if FLAGS.datasource == "continual_reacher":
                cum_energies = cum_energies + FLAGS.v_coeff * tf.reduce_mean(tf.square(x_joint[:, 1:, :, :-3] - x_joint[:, :-1, :, :-3]))
            else:
                cum_energies = cum_energies + FLAGS.v_coeff * tf.reduce_mean(tf.square(x_joint[:, 1:] - x_joint[:, :-1]))

        if FLAGS.constraint_accel:
            if FLAGS.datasource == "continual_reacher":
                cum_energies = cum_energies + FLAGS.a_coeff * tf.reduce_mean(tf.square(x_joint[:, 2:, :, :-3] - 2 * x_joint[:, 1:-1, :, :-3] + x_joint[:, :-2, :, :-3]))
            else:
                cum_energies = cum_energies + FLAGS.a_coeff * tf.reduce_mean(tf.square(x_joint[:, 2:] - 2 * x_joint[:, 1:-1] + x_joint[:, :-2]))
        # if FLAGS.constraint_goal:
        #     cum_energies = cum_energies + 0.1 * tf.reduce_mean(tf.square(x_joint - X_END))

        if FLAGS.constraint_goal:
            if FLAGS.datasource == "maze" or FLAGS.datasource == "point":
                goal_error = FLAGS.g_coeff * tf.reduce_mean(tf.square(x_joint - X_END), axis=[1, 2, 3])
                cum_energies = cum_energies + tf.expand_dims(goal_error, axis=1)
            elif FLAGS.datasource == "reacher":
                goal_error = FLAGS.g_coeff * tf.reduce_mean(tf.square(x_joint[:, :, :, :2] - X_END), axis=[1, 2, 3])
                cum_energies = cum_energies + tf.expand_dims(goal_error, axis=1)
            elif FLAGS.datasource == "continual_reacher":
                goal_error = FLAGS.g_coeff * tf.reduce_mean(tf.square(x_joint[:, :, :, -6:-3] - X_END), axis=[1, 2, 3])
                cum_energies = cum_energies + tf.expand_dims(goal_error, axis=1)

        # L2 distance goal specification
        # weight should be adjusted accordingly
        if FLAGS.datasource == "maze" or FLAGS.datasource == "point":
            cum_energies = cum_energies + FLAGS.l_coeff * tf.reduce_mean(tf.square(x_joint[:, -1:] - X_END))
        elif FLAGS.datasource == "reacher":
            cum_energies = cum_energies + FLAGS.l_coeff * tf.reduce_mean(tf.square(x_joint[:, -1:, :, :2] - X_END))
        elif FLAGS.datasource == "continual_reacher":
            cum_energies = cum_energies + FLAGS.l_coeff * tf.reduce_mean(tf.square(x_joint[:, -1:, :, -6:-3] - X_END))
        else:
            raise AssertionError("Unsupported data source")

        x_grad, action_grad = tf.gradients(cum_energies, [x_joint, actions])
        # x_grad = tf.Print(x_grad, [x_grad[0]], "x_grad value")
        x_joint = x_joint - FLAGS.step_lr * anneal_val * x_grad
        x_joint = tf.concat([X_START, x_joint[:, 1:FLAGS.plan_steps + 1]], axis=1)

        # if FLAGS.cond:
        #     actions = actions - FLAGS.step_lr * anneal_val * action_grad
        #     actions = tf.clip_by_value(actions, -1.0, 1.0)

        counter = counter + 1

        return counter, x_joint, actions


    def mppi_step(counter, x_joint, actions):
        x_joint = x_joint[:, 1:]
        x_joint_tile = tf.tile(tf.expand_dims(x_joint, axis=1), (1, FLAGS.noise_sim, 1, 1, 1))
        x_joint_shape = tf.shape(x_joint_tile)
        x_joint_tile = x_joint_tile

        # Best for particle experiments
        # sample = smooth_trajectory_tf(0.00001)

        # Best for continual reacher experiments
        sample = smooth_trajectory_tf(0.00001 * FLAGS.traj_scale)
        # sample = tf.Print(sample, [tf.reduce_max(sample), tf.reduce_mean(tf.abs(sample))], "sample info")

        sample = tf.reshape(sample, (FLAGS.num_env, FLAGS.noise_sim, FLAGS.latent_dim, FLAGS.plan_steps, 1))
        sample = tf.cast(tf.transpose(sample, (0, 1, 3, 4, 2)), tf.float32)
        x_joint_tile = x_joint_tile + sample
        # x_joint_tile = tf.Print(x_joint_tile, [tf.shape(sample), sample[0, 0, :, 0, 0]], "value of sample")
        x_joint = tf.reshape(x_joint_tile, (-1, x_joint_shape[2], x_joint_shape[3], x_joint_shape[4]))

        x_end_shape = tf.shape(X_END)
        X_END_EXPAND = tf.tile(tf.expand_dims(X_END, axis=1), (1, FLAGS.noise_sim, 1, 1, 1))
        X_END_EXPAND = tf.reshape(X_END_EXPAND, (-1, x_end_shape[1], x_end_shape[2], x_end_shape[3]))

        x_start_shape = tf.shape(X_START)
        X_START_EXPAND_TILE = tf.tile(tf.expand_dims(X_START, axis=1), (1, FLAGS.noise_sim, 1, 1, 1))
        X_START_EXPAND = tf.reshape(X_START_EXPAND_TILE, (-1, x_start_shape[1], x_start_shape[2], x_start_shape[3]))

        x_joint = tf.concat([X_START_EXPAND, x_joint], axis=1)
        x_joint_tile = tf.concat([X_START_EXPAND_TILE, x_joint_tile], axis=2)

        cum_energies = 0
        for i in range(FLAGS.plan_steps - FLAGS.total_frame + 2):
            if cond:
                cum_energy = model.forward(x_joint[:, i:i + FLAGS.total_frame], weights, action_label=actions[:, i], opt_low=FLAGS.opt_low)
            else:
                cum_energy = model.forward(x_joint[:, i:i + FLAGS.total_frame], weights, action_label=None, opt_low=FLAGS.opt_low)

            # if i == 0:
            #     cum_energy = tf.Print(cum_energy, [tf.pow(100 * cum_energy, 3)], "Energy on first timestep")

            # cum_energies = cum_energies + (FLAGS.plan_steps - i * 0.5 / FLAGS.plan_steps) ** 5 * tf.pow(cum_energy, 3)
            # cum_energies = cum_energies + tf.pow(cum_energy, 3)
            cum_energies = cum_energies + cum_energy

        # cum_energies = tf.Print(cum_energies, [x_joint[0, 0], X_START[0, 0]], "trajectory")

        # cum_energies = tf.Print(cum_energies, [cum_energies], "cumulative energies")
        if FLAGS.constraint_vel:
            cum_energies = cum_energies + l2_weight * FLAGS.v_coeff * tf.expand_dims(tf.reduce_mean(tf.square(x_joint[:, 1:] - x_joint[:, :-1]), axis=[1, 2, 3]), axis=1)

        if FLAGS.constraint_accel:
            cum_energies = cum_energies + l2_weight * FLAGS.a_coeff * tf.expand_dims(tf.reduce_mean(tf.square(x_joint[:, 2:] - 2 * x_joint[:, 1:-1] + x_joint[:, :-2]), axis=[1, 2, 3]), axis=1)


        # cum_energies = tf.Print(cum_energies, [tf.shape(cum_energies), tf.shape(X_END_EXPAND)], "cumulative energies")
        # score_weights = tf.Print(score_weights, [score_weights])

        scaler_weight = tf.square(tf.to_float(tf.reshape(tf.range(FLAGS.plan_steps+1) / FLAGS.plan_steps, (1, FLAGS.plan_steps + 1, 1, 1)))) * l2_weight
        # scaler_weight = tf.ones(1) * l2_weight
        if FLAGS.constraint_goal:
            if FLAGS.datasource == "maze" or FLAGS.datasource == "point":
                goal_error = l2_weight * FLAGS.g_coeff * tf.reduce_mean(tf.square(scaler_weight * (x_joint[:, :] - X_END_EXPAND)), axis=[1, 2, 3])
                cum_energies = cum_energies + tf.expand_dims(goal_error, axis=1)
            elif FLAGS.datasource == "reacher":
                goal_error = l2_weight * FLAGS.g_coeff * tf.reduce_mean(tf.square(scaler_weight * (x_joint[:, :, :, :2] - X_END_EXPAND)), axis=[1, 2, 3])
                cum_energies = cum_energies + tf.expand_dims(goal_error, axis=1)
            elif FLAGS.datasource == "continual_reacher":
                goal_error = l2_weight * FLAGS.g_coeff * tf.reduce_mean(tf.square(scaler_weight * (x_joint[:, :, :, -6:-3] - X_END_EXPAND)), axis=[1, 2, 3])
                cum_energies = cum_energies + tf.expand_dims(goal_error, axis=1)

        # X_END_EXPAND = tf.Print(X_END_EXPAND, [X_END_EXPAND[0], x_joint[0]], "X_END value")
        if FLAGS.datasource == "maze" or FLAGS.datasource == "point":
            cum_energies = cum_energies + FLAGS.l_coeff * tf.reduce_mean(tf.square(x_joint[:, -1:] - X_END_EXPAND))
        elif FLAGS.datasource == "reacher":
            cum_energies = cum_energies + FLAGS.l_coeff * tf.reduce_mean(tf.square(x_joint[:, -1:, :, :2] - X_END_EXPAND))
        elif FLAGS.datasource == "continual_reacher":
            cum_energies = cum_energies + FLAGS.l_coeff * tf.reduce_mean(tf.square(x_joint[:, -1:, :, -6:-3] - X_END_EXPAND))
        else:
            raise AssertionError("Unsupported data source")


        cum_energies = tf.reshape(cum_energies, (x_joint_shape[0], FLAGS.noise_sim))
        # cum_energies = tf.Print(cum_energies, [cum_energies], "cum_energies")
        score_weights = tf.nn.softmax(-cum_energies * 10000, axis=1)
        # score_weights = tf.Print(score_weights, [score_weights], "score_weights")
        score_weights = tf.reshape(score_weights, (x_joint_shape[0], FLAGS.noise_sim, 1, 1, 1))

        x_joint = tf.reduce_sum(x_joint_tile * score_weights, axis=1)
        # x_joint = tf.concat([X_START, x_joint[:, 1:FLAGS.plan_steps + 1]], axis=1)
        x_joint = x_joint[:, :FLAGS.plan_steps + 1]

        if FLAGS.datasource != "continual_reacher":
            x_joint = tf.clip_by_value(x_joint, -1, 1)

        counter = counter + 1

        return counter, x_joint, actions


    if FLAGS.mppi:
        steps, x_joint, actions = tf.while_loop(c, mppi_step, (steps, x_joint, actions))
    else:
        steps, x_joint, actions = tf.while_loop(c, mcmc_step, (steps, x_joint, actions))

    if FLAGS.gt_inverse_dynamics:
        actions = tf.clip_by_value(20.0 * (x_joint[:, 1:, 0] - x_joint[:, :-1, 0]), -1.0, 1.0)
    else:
        idyn_model = TrajInverseDynamics(dim_output=FLAGS.action_dim, dim_input=FLAGS.latent_dim)
        weights = idyn_model.construct_weights(scope="inverse_dynamics", weights=weights, reuse=tf.AUTO_REUSE)
        batch_size = tf.shape(x_joint)[0]
        pair_states = tf.concat([x_joint[:, i:i + 2] for i in range(FLAGS.plan_steps)], axis=0)
        actions = idyn_model.forward(pair_states, weights)
        actions = tf.transpose(tf.reshape(actions, (FLAGS.plan_steps, batch_size, FLAGS.action_dim)), (1, 0, 2))

    cum_energies = []
    for i in range(FLAGS.plan_steps - FLAGS.total_frame + 2):
        cum_energy = model.forward(x_joint[:, i:i + FLAGS.total_frame], weights, action_label=None)
        cum_energies.append(cum_energy)
        # cum_energies = tf.Print(cum_energies, [cum_energy], "Transition")

    cum_energies = tf.concat(cum_energies, axis=1)

    target_vars['actions'] = actions
    target_vars['x_joint'] = x_joint
    target_vars['cum_plan_energy'] = cum_energies
    target_vars['X_START'] = X_START
    target_vars['X_END'] = X_END
    target_vars['X_PLAN'] = X_PLAN
    target_vars['ACTION_PLAN'] = ACTION_LABEL

    return target_vars


def construct_eval_collision_model(model, weights, X_PLAN, target_vars={}):
    steps = tf.constant(0)
    MASK_PLAN = target_vars['mask_plan']

    c = lambda i, x: tf.less(i, FLAGS.num_steps)

    mask_plan_expand = tf.expand_dims(MASK_PLAN, axis=3)
    mask_plan_expand = tf.tile(tf.expand_dims(mask_plan_expand, axis=1), (1, FLAGS.noise_sim, 1, 1, 1))

    def mppi_step(counter, x_joint):
        x_joint_tile = tf.tile(tf.expand_dims(x_joint, axis=1), (1, FLAGS.noise_sim, 1, 1, 1))
        x_joint_shape = tf.shape(x_joint_tile)

        sample = smooth_trajectory_tf(0.00001 * FLAGS.traj_scale, FLAGS.batch_size)
        sample = tf.reshape(sample, (FLAGS.batch_size, FLAGS.noise_sim, FLAGS.latent_dim, FLAGS.plan_steps, 1))
        sample = tf.cast(tf.transpose(sample, (0, 1, 3, 4, 2)), tf.float32)
        x_joint_tile = x_joint_tile + sample * mask_plan_expand
        x_joint = tf.reshape(x_joint_tile, (-1, x_joint_shape[2], x_joint_shape[3], x_joint_shape[4]))

        cum_energies = 0

        for i in range(FLAGS.plan_steps - FLAGS.total_frame + 1):
            cum_energy = model.forward(x_joint[:, i:i + FLAGS.total_frame], weights)
            cum_energies = cum_energies + cum_energy

        cum_energies = tf.reshape(cum_energies, (x_joint_shape[0], FLAGS.noise_sim))
        score_weights = tf.nn.softmax(-cum_energies * 20000, axis=1)
        score_weights = tf.reshape(score_weights, (x_joint_shape[0], FLAGS.noise_sim, 1, 1, 1))
        x_joint = tf.reduce_sum(x_joint_tile * score_weights, axis=1)

        counter = counter + 1

        return counter, x_joint

    steps, x_joint = tf.while_loop(c, mppi_step, (steps, X_PLAN))

    cum_energies = []
    for i in range(FLAGS.plan_steps - FLAGS.total_frame + 1):
        cum_energy = model.forward(x_joint[:, i:i + FLAGS.total_frame], weights, action_label=None)
        cum_energies.append(cum_energy)
        # cum_energies = tf.Print(cum_energies, [cum_energy], "Transition")

    cum_energies = tf.concat(cum_energies, axis=1)

    target_vars['x_joint'] = tf.expand_dims(x_joint, axis=2)
    target_vars['cum_energies'] = cum_energies
    target_vars['X_PLAN'] = X_PLAN

    return target_vars


def construct_ff_plan_model(model, weights, X_PLAN, X_START, X_END, ACTION_PLAN, target_vars={}):
    actions = ACTION_PLAN
    steps = tf.constant(0)
    l2_weight = tf.placeholder(shape=None, dtype=tf.float32)
    num_steps = tf.placeholder(shape=None, dtype=tf.float32)
    target_vars['l2_weight'] = l2_weight
    target_vars['num_steps'] = num_steps

    print("here!??!")
    c = lambda i, x, y: tf.less(i, FLAGS.num_steps)

    def mcmc_step(counter, actions, x_vals):
        actions = actions + tf.random_normal(tf.shape(actions), mean=0.0, stddev=0.01)

        x_val = X_START
        x_vals = []
        for i in range(FLAGS.plan_steps):
            x_val = model.forward(x_val, weights, action_label=actions[:, i])
            x_vals.append(x_val)

        x_vals = tf.stack(x_vals, axis=1)

        if FLAGS.anneal:
            anneal_val = tf.cast(counter, tf.float32) / FLAGS.num_steps
        else:
            anneal_val = 1

        if FLAGS.datasource == "reacher":
            energy = tf.reduce_sum(tf.square(x_val[:, :2] - X_END[:, 0, 0]))
        elif FLAGS.datasource == "continual_reacher":
            energy = tf.reduce_sum(tf.square(x_val[:, -6:-3] - X_END[:, 0, 0]))
        else:
            energy = tf.reduce_sum(tf.square(x_val - X_END[:, 0, 0]))

        action_grad = tf.gradients(energy, [actions])[0]
        actions = actions - FLAGS.step_lr * anneal_val * action_grad
        actions = tf.clip_by_value(actions, -1.0, 1.0)

        counter = counter + 1

        return counter, actions, x_vals

    steps, actions, x_joint = tf.while_loop(c, mcmc_step,
                                            (steps, actions, X_PLAN[:, :, 0]))

    target_vars['x_joint'] = tf.expand_dims(x_joint, axis=2)
    target_vars['actions'] = actions
    target_vars['X_START'] = X_START
    target_vars['X_END'] = X_END
    target_vars['X_PLAN'] = X_PLAN
    target_vars['ACTION_PLAN'] = ACTION_PLAN
    target_vars['cum_plan_energy'] = tf.zeros(1)

    return target_vars


def construct_ff_model(model, weights, X_NOISE, X, ACTION_LABEL, ACTION_NOISE_LABEL, optimizer, target_vars={}):
    x_mods = []

    x_pred = model.forward(X[:, 0, 0], weights, action_label=ACTION_LABEL)
    loss_total = tf.reduce_mean(tf.square(x_pred - X[:, 1, 0]))

    dyn_model = TrajInverseDynamics(dim_input=FLAGS.latent_dim, dim_output=FLAGS.action_dim)
    weights = dyn_model.construct_weights(scope="inverse_dynamics", weights=weights)
    output_action = dyn_model.forward(X, weights)
    dyn_loss = tf.reduce_mean(tf.square(output_action - ACTION_LABEL))
    dyn_dist = tf.reduce_mean(tf.abs(output_action - ACTION_LABEL))
    target_vars['dyn_loss'] = dyn_loss
    target_vars['dyn_dist'] = dyn_dist

    dyn_optimizer = AdamOptimizer(1e-4)
    gvs = dyn_optimizer.compute_gradients(dyn_loss)
    dyn_train_op = dyn_optimizer.apply_gradients(gvs)

    gvs = optimizer.compute_gradients(loss_total)
    gvs = [(k, v) for (k, v) in gvs if k is not None]
    print("Applying gradients...")
    grads, vs = zip(*gvs)

    def filter_grad(g, v):
        return tf.clip_by_value(g, -1e5, 1e5)

    capped_gvs = [(filter_grad(grad, var), var) for grad, var in gvs]
    gvs = capped_gvs
    train_op = optimizer.apply_gradients(gvs)

    if not FLAGS.gt_inverse_dynamics:
        train_op = tf.group(train_op, dyn_train_op)

    target_vars['train_op'] = train_op

    target_vars['loss_ml'] = tf.zeros(1)
    target_vars['total_loss'] = loss_total
    target_vars['gvs'] = gvs
    target_vars['loss_energy'] = tf.zeros(1)
    target_vars['ff_loss'] = loss_total
    target_vars['ff_dist'] = tf.zeros(1)

    target_vars['weights'] = weights
    target_vars['X'] = X
    target_vars['X_NOISE'] = X_NOISE
    target_vars['energy_pos'] = tf.zeros(1)
    target_vars['energy_neg'] = tf.zeros(1)
    target_vars['x_grad'] = tf.zeros(1)
    target_vars['action_grad'] = tf.zeros(1)
    target_vars['x_mod'] = tf.zeros(1)
    target_vars['x_off'] = tf.zeros(1)
    target_vars['temp'] = FLAGS.temperature
    target_vars['ACTION_LABEL'] = ACTION_LABEL
    target_vars['ACTION_NOISE_LABEL'] = ACTION_NOISE_LABEL
    target_vars['progress_diff'] = tf.zeros(1)

    return target_vars


def construct_model(model, weights, X_NOISE, X, ACTION_LABEL, ACTION_NOISE_LABEL, LR, optimizer, target_vars={}):
    x_mods = []

    energy_noise = model.forward(X_NOISE, weights, reuse=True, stop_at_grad=True, action_label=ACTION_LABEL)

    print("Building graph...")
    x_mod = X_NOISE
    MASK = target_vars['mask']

    mask_expand = mask_expand_single = tf.expand_dims(MASK, axis=3)

    x_grads = []
    x_ees = []
    energy_negs = [energy_noise]
    loss_energys = []

    if not FLAGS.gt_inverse_dynamics:
        dyn_model = TrajInverseDynamics(dim_input=FLAGS.latent_dim, dim_output=FLAGS.action_dim)
        weights = dyn_model.construct_weights(scope="inverse_dynamics", weights=weights)

    if FLAGS.ff_model:
        ff_model = TrajFFDynamics(dim_input=FLAGS.latent_dim, dim_output=FLAGS.latent_dim)
        weights = ff_model.construct_weights(scope="ff_model", weights=weights)

    steps = tf.constant(0)
    steps_pos = tf.constant(0)
    c = lambda i, x, y: tf.less(i, FLAGS.num_steps)

    def mcmc_step(counter, x_mod, action_label):
        x_mod = x_mod + tf.random_normal(tf.shape(x_mod), mean=0.0, stddev=0.01)
        action_label = action_label + tf.random_normal(tf.shape(action_label), mean=0.0, stddev=0.01)

        energy_noise = model.forward(x_mod, weights, action_label=action_label, reuse=True, stop_at_grad=True)
        lr = FLAGS.step_lr

        x_grad = tf.gradients(FLAGS.temperature * energy_noise, [x_mod])[0]
        x_grad = tf.concat([tf.zeros(tf.shape(x_grad[:, :-1])), x_grad[:, -1:]], axis=1)

        x_mod = x_mod - lr * x_grad

        if FLAGS.cond:
            x_grad, action_grad = tf.gradients(FLAGS.temperature * energy_noise, [x_mod, action_label])
        else:
            x_grad, action_grad = tf.gradients(FLAGS.temperature * energy_noise, [x_mod])[0], tf.zeros(1)

        action_label = action_label - FLAGS.step_lr * action_grad

        counter = counter + 1

        return counter, x_mod, action_label

    mask_expand = tf.tile(tf.expand_dims(mask_expand, axis=1), (1, FLAGS.noise_sim, 1, 1, 1))

    def mppi_step(counter, x_mod, action_label):
        x_mod_tile = tf.tile(tf.expand_dims(x_mod, axis=1), (1, FLAGS.noise_sim, 1, 1, 1))
        x_mod_shape = tf.shape(x_mod_tile)

        x_mod_tile = x_mod_tile + tf.random_normal(tf.shape(x_mod_tile), mean=0.0, stddev=0.01 * FLAGS.traj_scale)

        x_mod = tf.reshape(x_mod_tile, (-1, x_mod_shape[2], x_mod_shape[3], x_mod_shape[4]))

        energy_noise = model.forward(x_mod, weights, action_label=action_label, reuse=True, stop_at_grad=True)

        energy_noise = tf.reshape(energy_noise, (x_mod_shape[0], FLAGS.noise_sim))
        score_weights = tf.nn.softmax(-energy_noise*50000, axis=1)
        score_weights = tf.reshape(score_weights, (x_mod_shape[0], FLAGS.noise_sim, 1, 1, 1))
        x_mod = tf.reduce_sum(x_mod_tile * score_weights, axis=1)

        counter = counter + 1

        return counter, x_mod, action_label


    def mppi_step_mask(counter, x_mod, action_label):
        x_mod_tile = tf.tile(tf.expand_dims(x_mod, axis=1), (1, FLAGS.noise_sim, 1, 1, 1))
        x_mod_shape = tf.shape(x_mod_tile)

        x_mod_tile = x_mod_tile + tf.random_normal(tf.shape(x_mod_tile), mean=0.0, stddev=0.01) * mask_expand

        x_mod = tf.reshape(x_mod_tile, (-1, x_mod_shape[2], x_mod_shape[3], x_mod_shape[4]))

        energy_noise = model.forward(x_mod, weights, action_label=action_label, reuse=True, stop_at_grad=True)

        energy_noise = tf.reshape(energy_noise, (x_mod_shape[0], FLAGS.noise_sim))
        score_weights = tf.nn.softmax(-energy_noise*50000, axis=1)
        score_weights = tf.reshape(score_weights, (x_mod_shape[0], FLAGS.noise_sim, 1, 1, 1))
        x_mod = tf.reduce_sum(x_mod_tile * score_weights, axis=1)

        counter = counter + 1

        return counter, x_mod, action_label


    def mcmc_step_mask(counter, x_mod, action_label):
        x_mod = x_mod + tf.random_normal(tf.shape(x_mod), mean=0.0, stddev=0.01) * mask_expand_single
        energy_noise = model.forward(x_mod, weights, action_label=action_label, reuse=True, stop_at_grad=True)
        lr = FLAGS.step_lr

        x_grad = tf.gradients(FLAGS.temperature * energy_noise, [x_mod])[0]

        x_mod = x_mod - lr * x_grad * mask_expand_single

        counter = counter + 1

        return counter, x_mod, action_label


    if FLAGS.mppi:
        steps, x_mod, action_label = tf.while_loop(c, mppi_step, (steps, x_mod, ACTION_NOISE_LABEL))
    else:
        steps, x_mod, action_label = tf.while_loop(c, mcmc_step, (steps, x_mod, ACTION_NOISE_LABEL))

    if FLAGS.datasource == "collision":
        c_collision = lambda i, x, y: tf.less(i, FLAGS.num_steps)
        if FLAGS.mppi:
            steps, X_new, action_label = tf.while_loop(c_collision, mppi_step_mask, (steps_pos, X, ACTION_NOISE_LABEL))
        else:
            steps, X_new, action_label = tf.while_loop(c_collision, mcmc_step_mask, (steps_pos, X, ACTION_NOISE_LABEL))

        energy_pos = model.forward(tf.stop_gradient(X_new), weights, action_label=ACTION_LABEL)
        x_off = tf.reduce_mean(tf.abs(X_new - X))
    else:
        energy_pos = model.forward(X, weights, action_label=ACTION_LABEL)
        x_off = tf.reduce_mean(tf.abs(x_mod - X))

    if FLAGS.cond:
        if FLAGS.datasource != "reacher":
            progress_diff = tf.reduce_mean(tf.abs((x_mod[:, 1, 0] - x_mod[:, 0, 0]) - action_label / 20))
        else:
            progress_diff = tf.zeros(1)
    else:
        progress_diff = tf.zeros(1)

    target_vars['x_mod'] = x_mod
    temp = FLAGS.temperature

    loss_energy = temp * model.forward(x_mod, weights, reuse=True, action_label=action_label, stop_grad=True)
    x_mod = tf.stop_gradient(x_mod)
    action_label = tf.stop_gradient(action_label)

    energy_neg = model.forward(x_mod, weights, action_label=action_label, reuse=True)
    # energy_neg = tf.Print(energy_neg, [energy_neg], message="energy_neg weights")
    if FLAGS.cond:
        x_grad, action_grad = tf.gradients(FLAGS.temperature * energy_neg, [x_mod, action_label])
    else:
        x_grad, action_grad = tf.gradients(FLAGS.temperature * energy_neg, [x_mod])[0], tf.zeros(1)

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
            # weight_coeff = tf.stop_gradient(tf.nn.softmax(-0.01 * energy_pos, axis=0))
            weight_coeff = tf.ones(1)
            inverse_sum = tf.reduce_sum(1. / weight_coeff)
            neg_loss = -tf.reduce_mean(temp * energy_neg)
            loss_ml = FLAGS.ml_coeff * (tf.reduce_mean(pos_loss / weight_coeff) + inverse_sum * tf.reduce_sum(neg_loss))
        elif FLAGS.objective == 'softplus':
            loss_ml = FLAGS.ml_coeff * \
                      tf.nn.softplus(temp * (energy_pos - energy_neg))

        loss_total = tf.reduce_mean(loss_ml)

        if not FLAGS.zero_kl:
            loss_total = loss_total + tf.reduce_mean(loss_energy)

        loss_total = loss_total + \
                     FLAGS.l2_coeff * (tf.reduce_mean(tf.square(energy_pos)) + tf.reduce_mean(tf.square((energy_neg))))

    if not FLAGS.gt_inverse_dynamics:
        output_action = dyn_model.forward(X, weights)
        dyn_loss = tf.reduce_mean(tf.square(output_action - ACTION_LABEL))
        dyn_dist = tf.reduce_mean(tf.abs(output_action - ACTION_LABEL))
        target_vars['dyn_loss'] = dyn_loss
        target_vars['dyn_dist'] = dyn_dist

        dyn_optimizer = AdamOptimizer(1e-3)
        gvs = dyn_optimizer.compute_gradients(dyn_loss)
        dyn_train_op = dyn_optimizer.apply_gradients(gvs)
    else:
        target_vars['dyn_loss'] = tf.zeros(1)
        target_vars['dyn_dist'] = tf.zeros(1)

    if FLAGS.ff_model:
        # This input must be a single state-state transition
        assert (FLAGS.total_frame == 2)
        X_LABEL = X[:, -1, 0]
        output_x = ff_model.forward(X[:, 0], weights, action_label=ACTION_LABEL)

        if FLAGS.datasource == "reacher":
            ff_loss = tf.reduce_mean(tf.square(output_x - X_LABEL))
            ff_dist = tf.reduce_mean(tf.abs(output_x - X_LABEL))
        else:
            ff_loss = tf.reduce_mean(tf.square(output_x - X_LABEL))
            ff_dist = tf.reduce_mean(tf.abs(output_x - X_LABEL))
        target_vars['ff_loss'] = ff_loss
        target_vars['ff_dist'] = ff_dist

        ff_optimizer = AdamOptimizer(1e-3)
        gvs = ff_optimizer.compute_gradients(ff_loss)
        ff_train_op = ff_optimizer.apply_gradients(gvs)

    else:
        target_vars['ff_loss'] = tf.zeros(1)
        target_vars['ff_dist'] = tf.zeros(1)

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

        train_ops = [train_op]
        if not FLAGS.gt_inverse_dynamics:
            train_ops.append(dyn_train_op)
        if FLAGS.ff_model:
            train_ops.append(ff_train_op)

        train_op = tf.group(*train_ops)

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
    target_vars['action_grad'] = action_grad
    target_vars['x_mod'] = x_mod
    target_vars['x_off'] = x_off
    target_vars['temp'] = temp
    target_vars['lr'] = LR
    target_vars['ACTION_LABEL'] = ACTION_LABEL
    target_vars['ACTION_NOISE_LABEL'] = ACTION_NOISE_LABEL
    target_vars['progress_diff'] = progress_diff

    return target_vars


def main():
    logdir = osp.join(FLAGS.logdir, FLAGS.exp)
    if not osp.exists(logdir):
        os.makedirs(logdir)
    logger = TensorBoardOutputFormat(logdir)

    if FLAGS.datasource == 'point' or FLAGS.datasource == 'maze' or FLAGS.datasource == 'reacher' or \
            FLAGS.datasource == 'phy_a' or FLAGS.datasource == 'phy_cor' or FLAGS.datasource == "continual_reacher" or FLAGS.datasource == "collision":
        if FLAGS.ff_model:
            model = TrajFFDynamics(dim_input=FLAGS.latent_dim, dim_output=FLAGS.latent_dim)
        else:
            model = TrajNetLatentFC(dim_input=FLAGS.latent_dim)

        X_NOISE = tf.placeholder(shape=(None, FLAGS.total_frame, FLAGS.input_objects, FLAGS.latent_dim),
                                 dtype=tf.float32)
        X = tf.placeholder(shape=(None, FLAGS.total_frame, FLAGS.input_objects, FLAGS.latent_dim), dtype=tf.float32)

        if FLAGS.datasource == "collision":
            ACTION_LABEL = tf.placeholder(shape=(None, FLAGS.input_objects, FLAGS.latent_dim), dtype=tf.float32)
            ACTION_NOISE_LABEL = tf.placeholder(shape=(None, FLAGS.action_dim), dtype=tf.float32)
        else:
            ACTION_LABEL = tf.placeholder(shape=(None, FLAGS.action_dim), dtype=tf.float32)
            ACTION_NOISE_LABEL = tf.placeholder(shape=(None, FLAGS.action_dim), dtype=tf.float32)
        ACTION_PLAN = tf.placeholder(shape=(None, FLAGS.plan_steps, FLAGS.action_dim), dtype=tf.float32)

        X_START = tf.placeholder(shape=(None, 1, FLAGS.input_objects, FLAGS.latent_dim), dtype=tf.float32)
        X_PLAN = tf.placeholder(shape=(None, FLAGS.plan_steps, FLAGS.input_objects, FLAGS.latent_dim), dtype=tf.float32)

        MASK = tf.placeholder(shape=(None, FLAGS.total_frame, FLAGS.input_objects), dtype=tf.float32)
        MASK_PLAN = tf.placeholder(shape=(None, FLAGS.plan_steps, FLAGS.input_objects), dtype=tf.float32)

        if FLAGS.datasource == 'reacher':
            X_END = tf.placeholder(shape=(None, 1, FLAGS.input_objects, 2), dtype=tf.float32)
        elif FLAGS.datasource == 'continual_reacher':
            X_END = tf.placeholder(shape=(None, 1, FLAGS.input_objects, 3), dtype=tf.float32)
        else:
            X_END = tf.placeholder(shape=(None, 1, FLAGS.input_objects, FLAGS.latent_dim), dtype=tf.float32)
    else:
        raise AssertionError("Unsupported data source")

    if FLAGS.ff_model and FLAGS.pretrain_eval:
        weights = model.construct_weights(action_size=FLAGS.action_dim, scope="ff_model")
    else:
        weights = model.construct_weights(action_size=FLAGS.action_dim)

    LR = tf.placeholder(shape=(), dtype=tf.float32)
    optimizer = AdamOptimizer(LR, beta1=0.0, beta2=0.999)

    target_vars = {}
    target_vars['lr'] = LR
    target_vars['mask'] = MASK
    target_vars['mask_plan'] = MASK_PLAN


    if FLAGS.train or FLAGS.debug:
        if FLAGS.ff_model:
            target_vars = construct_ff_model(model, weights, X_NOISE, X, ACTION_LABEL, ACTION_NOISE_LABEL, optimizer, target_vars=target_vars)
        else:
            target_vars = construct_model(model, weights, X_NOISE, X, ACTION_LABEL, ACTION_NOISE_LABEL, LR, optimizer, target_vars=target_vars)

    print(target_vars.keys())

    if FLAGS.eval_collision:
        target_vars = construct_eval_collision_model(model, weights, X_PLAN, target_vars=target_vars)
    elif not FLAGS.train or FLAGS.rl_train:
        # evaluation
        if FLAGS.ff_model:
            target_vars = construct_ff_plan_model(model, weights, X_PLAN, X_START, X_END, ACTION_PLAN, target_vars=target_vars)
        else:
            target_vars = construct_plan_model(model, weights, X_PLAN, X_START, X_END, ACTION_PLAN, FLAGS.cond, target_vars=target_vars)

    print(target_vars.keys())
    sess = tf.InteractiveSession()
    saver = tf.train.Saver(max_to_keep=10, keep_checkpoint_every_n_hours=2)

    # count number of parameters in model
    total_parameters = 0
    for variable in tf.trainable_variables():
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        total_parameters += variable_parameters

    print("Model has a total of {} parameters".format(total_parameters))

    tf.global_variables_initializer().run()
    print("Initializing variables...")

    resume_itr = 0

    if FLAGS.resume_iter != -1 or not FLAGS.train:
        model_file = osp.join(logdir, 'model_{}'.format(FLAGS.resume_iter))
        resume_itr = FLAGS.resume_iter
        saver.restore(sess, model_file)

    if FLAGS.rl_train:

        datasource = FLAGS.datasource
        def make_env(rank):
            def _thunk():
                # Make the environments non stoppable for now
                if datasource == "maze":
                    env = Maze(end=[0.7, -0.8], start=[-0.85, -0.85], random_starts=False)
                elif datasource == "point":
                    env = Point(end=[0.5, 0.5], start=[0.0, 0.0], random_starts=True)
                elif datasource == "reacher":
                    env = Reacher(end=[0.7, 0.5], eps=0.01)
                elif datasource == "continual_reacher":
                    env = ContinualReacher7DOFEnv()
                env.seed(rank)
                env = Monitor(env, os.path.join("/tmp", str(rank)), allow_early_resets=True)
                return env

            return _thunk

        env = SubprocVecEnv([make_env(i + FLAGS.seed) for i in range(FLAGS.num_env)])

        rl_train(target_vars, saver, sess, logger, FLAGS.resume_iter, env)

    else:
        if FLAGS.n_benchmark_exp > 0:
            # perform benchmark experiments (originally in pipeline.py)
            start_arr = [FLAGS.start1, FLAGS.start2]
            end_arr = [FLAGS.end1, FLAGS.end2]

            if FLAGS.datasource == 'point':
                env = Point(start_arr, end_arr, FLAGS.eps, FLAGS.obstacle)
            elif FLAGS.datasource == 'maze':
                env = Maze(start_arr, end_arr, FLAGS.eps, FLAGS.obstacle)
            elif FLAGS.datasource == 'reacher':
                env = Reacher([0.7, 0.5], FLAGS.eps)
            elif FLAGS.datasource == 'phy_a':
                env = Ball(a=[0.05, 0.05], random_starts=True, eps=FLAGS.eps)
            elif FLAGS.datasource == 'phy_cor':
                env = Ball(cor=0.5, random_starts=True, eps=FLAGS.eps)
            elif FLAGS.datasource == 'continual_reacher':
                env = ContinualReacher7DOFEnv(train=False)
            else:
                raise KeyError

            get_avg_step_num(target_vars, sess, env)

        else:
            mask = None
            if FLAGS.continual_reacher_model:
                data = np.load(FLAGS.datadir + 'continual_reacher_model.npz')
                dataset = data['ob'][:, :, None, :]
                actions = data['action']
                actions = np.tile(actions[:, None, :], (1, 2, 1))
            elif FLAGS.datasource == "collision":
                data = np.load(FLAGS.datadir + 'collision.npz')
                dataset = data['arr_0']
                mask = data['arr_1']
                actions = data['arr_0']
                # Mask out that portion of the dataset
                dataset_old = dataset

                dataset = dataset * (1 - mask[:, :, :, None])
            else:
                data = np.load(FLAGS.datadir + FLAGS.datasource + '.npz')
                dataset = data['obs'][:, :, None, :]
                actions = data['action']

            if FLAGS.datasource == 'point' or FLAGS.datasource == 'maze' \
                    or FLAGS.datasource == 'phy_a' or FLAGS.datasource == 'phy_cor':
                mean, std = 0, 1
            elif FLAGS.datasource == 'reacher':
                dones = data['action']

                dataset[:, :, :, :2] = dataset[:, :, :, :2] % (2 * np.pi)
                s = dataset.shape

                dataset[:, :, :, :2] = (dataset[:, :, :, :2] - np.pi) / np.pi
                dataset[:, :, :, 2:] = dataset[:, :, :, 2:] / 10.0

                # dataset_flat = dataset.reshape((-1, FLAGS.latent_dim))
                # dataset = dataset / 55.
                # mean, std = dataset_flat.mean(axis=0), dataset_flat.std(axis=0)
                # std = std + 1e-5
                # dataset = (dataset - mean) / std
                print(dataset.max(), dataset.min())

                # For now a hacky way to deal with dones since each episode is always of length 50
                dataset = np.concatenate([dataset[:, 49:99], dataset[:, [99] + list(range(49))]], axis=0)
                actions = np.concatenate([actions[:, 49:99], actions[:, [99] + list(range(49))]], axis=0)
            elif FLAGS.datasource == 'continual_reacher':
                mean, std = 0, 1

            if FLAGS.single_task:
                # train on a single task
                dataset = np.tile(dataset[0:1], (100, 1, 1, 1))[:, :20]

            if FLAGS.datasource == 'phy_a' or FLAGS.datasource == 'phy_cor':
                if FLAGS.phy_latent:
                    # change physics param input to random noise
                    dataset[:, :, -1] = np.random.uniform(-1, 1)

            split_idx = int(dataset.shape[0] * 0.9)

            dataset_train = dataset[:split_idx]
            actions_train = actions[:split_idx]
            dataset_test = dataset[split_idx:]
            actions_test = actions[split_idx:]

            if FLAGS.train:
                train(target_vars, saver, sess, logger, dataset_train, actions_train, resume_itr, mask)

            if FLAGS.debug:
                debug(target_vars, sess)

            if FLAGS.eval_collision:
                dataset_test_gt = dataset_old[split_idx:]
                mask_test = mask[split_idx:]
                eval_collision(target_vars, sess, dataset_test, mask_test, dataset_test_gt)
            elif not FLAGS.train:
                test(target_vars, saver, sess, logdir, dataset_test, actions_test, dataset_train, mean, std)


if __name__ == "__main__":
    main()
