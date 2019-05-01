"""
A generic file to
(1) take a trained model for the specified environment
(2) run cond/no_cond benchmark
"""

import datetime
import os
import os.path as osp

import matplotlib as mpl
import matplotlib.patches as patches
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
from gen_data import is_maze_valid

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
flags.DEFINE_bool('anneal', False, 'Whether to use simulated annealing for sampling')

# Number of benchmark experiments
flags.DEFINE_integer('n_benchmark_exp', 0, 'Number of benchmark experiments')
flags.DEFINE_float('start1', 0.0, 'x_start, x')
flags.DEFINE_float('start2', 0.0, 'x_start, y')
flags.DEFINE_float('end1', 0.5, 'x_end, x')
flags.DEFINE_float('end2', 0.5, 'x_end, y')
flags.DEFINE_float('eps', 0.01, 'epsilon for done condition')
flags.DEFINE_list('obstacle', None, 'a size 4 array specifying top left and bottom right, e.g. [0.25, 0.35, 0.3, 0.3]')

# Additional constraints
flags.DEFINE_bool('constraint_vel', False, 'A distance constraint between each subsequent state')
flags.DEFINE_bool('constraint_goal', False, 'A distance constraint between current state and goal state')

flags.DEFINE_bool('debug', False, 'Print out energies when planning')
flags.DEFINE_bool('gt_inverse_dynamics', True, 'Whether to train a inverse dynamics model')
flags.DEFINE_bool('inverse_dynamics', False, 'Whether to train a inverse dynamics model')

flags.DEFINE_bool('save_single', False, 'Save every single trajectory')

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
        fieldnames = ['ts', 'start', 'actual_end', 'end', 'obstacle', 'plan_steps', 'cond', 'step_num', 'exp', 'iter']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writerow(d)


def get_avg_step_num(target_vars, sess, env):
    n_exp = FLAGS.n_benchmark_exp
    cond = 'True' if FLAGS.cond else 'False'
    obs = env.reset()
    start = obs
    collected_trajs = []

    for i in range(n_exp):
        points = []
        length = 0
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
            x_plan = np.random.uniform(-1, 1, (1, FLAGS.plan_steps, 1, 2))

            if FLAGS.cond:
                ACTION_PLAN = target_vars['ACTION_PLAN']
                actions = np.random.uniform(-0.05, 0.05, (1, FLAGS.plan_steps + 1, 2))
                x_joint, actions = sess.run([x_joint, output_actions],
                                                   {X_START: x_start, X_END: x_end,
                                                    X_PLAN: x_plan, ACTION_PLAN: actions})
            else:
                x_joint, output_actions = sess.run([x_joint, output_actions],
                                                   {X_START: x_start, X_END: x_end, X_PLAN: x_plan})
                # output_actions = output_actions[None, :, :]

            kill = False

            if FLAGS.cond:
                for i in range(actions.shape[1]):
                    obs, _, done, _ = env.step(actions[0, i, :])
                    target_obs = x_joint[0, i+1, 0]

                    print("obs", obs)
                    print("actions", actions[0, i, :])
                    print("target_obs", target_obs)
                    print("done?", done)
                    points.append(obs)

                    if done:
                        kill = True
                        break

                    if np.abs(target_obs - obs).mean() > 0.15:
                        break

            else:
                for i in range(output_actions.shape[1]):
                    obs, _, done, _ = env.step(output_actions[0, i, :])
                    target_obs = x_joint[0, i+1, 0]

                    print("obs", obs)
                    print("actions", output_actions[0, i, :])
                    print("target_obs", target_obs)
                    print("done?", done)
                    points.append(obs)

                    if done:
                        kill = True
                        break

                    if np.abs(target_obs - obs).mean() > 0.15:
                        break

            print("done")

            if kill:
                break

            if length > 10000:
                break

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
                ax = plt.gca()   # get the current reference
                rect = patches.Rectangle(xy, w, h, linewidth=1, edgecolor='r', facecolor='none')
                ax.add_patch(rect)
            elif FLAGS.datasource == "maze":
                # Plot the values of boundaries of the maze
                samples = np.random.uniform(-1, 1, (100000, 2))
                ob_mask = ~is_maze_valid(samples)
                walls = samples[ob_mask]
                plt.plot(walls[:, 0], walls[:, 1], 'ko')

        plt.plot(traj[:, 0], traj[:, 1])

        if FLAGS.save_single:
            save_dir = osp.join(imgdir, 'benchmark_{}_{}_iter{}_{}.png'.format(FLAGS.n_benchmark_exp, FLAGS.exp,
                                                                               FLAGS.resume_iter, timestamp))
            plt.savefig(save_dir)
            plt.clf()

        # save all length for calculation of average length
        lengths.append(traj.shape[0])

    if not FLAGS.save_single:
        save_dir = osp.join(imgdir, 'benchmark_{}_{}_iter{}_{}.png'.format(FLAGS.n_benchmark_exp, FLAGS.exp,
                                                                           FLAGS.resume_iter, timestamp))
        plt.savefig(save_dir)

    average_length = sum(lengths) / len(lengths)
    print("average number of steps:", average_length)


def construct_no_cond_plan_model(model, weights, X_PLAN, X_START, X_END, ACTION_LABEL):
    x_joint = tf.concat([X_START, X_PLAN], axis=1)
    steps = tf.constant(0)
    c = lambda i, x: tf.less(i, FLAGS.num_steps)


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
            # Sequential planning is  missing a goal specification right now
            assert False

            for i in range(FLAGS.plan_steps - FLAGS.total_frame + 2):
                x_temp = x_joint[:, i:i + FLAGS.total_frame]
                x_temp = x_temp + tf.random_normal(tf.shape(x_temp), mean=0.0, stddev=0.1)
                cum_energy = model.forward(x_temp, weights, action_label=ACTION_LABEL)
                x_grad = tf.gradients(cum_energy, [x_temp])[0]
                x_new = x_joint[:, i:i + FLAGS.total_frame] - FLAGS.step_lr * tf.cast(counter,
                                                                                      tf.float32) / FLAGS.num_steps * x_grad

                x_joint = tf.concat([x_joint[:, :i], x_new, x_joint[:, i + FLAGS.total_frame:]], axis=1)

                cum_energies.append(cum_energy)

            x_joint = tf.concat([X_START, x_joint[:, 1:FLAGS.plan_steps + 1]], axis=1)

            for i in range(FLAGS.plan_steps - FLAGS.total_frame + 1, -1):
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
            for i in range(FLAGS.plan_steps - FLAGS.total_frame + 2):
                x_temp = x_joint[:, i:i + FLAGS.total_frame]
                cum_energy = model.forward(x_temp, weights, action_label=ACTION_LABEL)
                cum_energies.append(cum_energy)

            cum_energies = tf.reduce_sum(tf.concat(cum_energies, axis=1), axis=1)

            if FLAGS.debug:
                cum_energies = tf.Print(cum_energies, [tf.reduce_mean(cum_energies)])

            if FLAGS.constraint_vel:
                cum_energies = cum_energies + 0.1 * tf.reduce_sum(tf.square((x_joint[:, 1:] - x_joint[:, :-1])))

            if FLAGS.constraint_goal:
                d = tf.reduce_sum(tf.square(x_joint - X_END))
                cum_energies =  cum_energies + d

            # TODO change to be the appropriate weight for distance to goal
            cum_energies = cum_energies + 1e-3 * tf.reduce_sum(tf.abs(x_joint[:, -1:] - X_END))

            x_grad = tf.gradients(cum_energies, [x_joint])[0]
            x_joint = x_joint - FLAGS.step_lr * tf.cast(counter, tf.float32) / FLAGS.num_steps * x_grad

        # Reset the start and end states to be previous values
        x_joint = tf.concat([X_START, x_joint[:, 1:FLAGS.plan_steps + 1]], axis=1)
        counter = counter + 1

        # counter = tf.Print(counter,
        #                    [tf.reduce_mean(cum_energies), tf.reduce_max(cum_energies), tf.reduce_min(cum_energies)])
        x_joint = tf.clip_by_value(x_joint, -1.0, 1.0)

        return counter, x_joint

    steps, x_joint = tf.while_loop(c, mcmc_step, (steps, x_joint))
    print("X_joint shape ", x_joint.get_shape())

    if FLAGS.gt_inverse_dynamics:
        actions = tf.clip_by_value(20.0 * (x_joint[:, 1:, 0] - x_joint[:, :-1, 0]), -1.0, 1.0)
    elif FLAGS.inverse_dynamics:
        idyn_model = TrajInverseDynamics()
        weights = idyn_model.construct_weights(scope="inverse_dynamics", weights=weights)
        pair_states = tf.concat([x_joint[:, i:i+2] for i in range(FLAGS.plan_steps+1)], axis=0)
        actions = idyn_model.forward(pair_states, weights)

    print("actions shape ", actions.get_shape())
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
    x_joint = tf.concat([X_START, X_PLAN], axis=1)
    steps = tf.constant(0)
    c = lambda i, x, y: tf.less(i, FLAGS.num_steps)

    def mcmc_step(counter, x_joint, actions):
        actions = actions + tf.random_normal(tf.shape(actions), mean=0.0, stddev=0.01)
        x_joint = x_joint + tf.random_normal(tf.shape(x_joint), mean=0.0, stddev=0.01)
        cum_energies = 0
        for i in range(FLAGS.plan_steps - FLAGS.total_frame + 3):
            cum_energy = model.forward(x_joint[:, i:i + FLAGS.total_frame], weights, action_label=actions[:, i])
            cum_energies = cum_energies + cum_energy

        if FLAGS.anneal:
            anneal_const = tf.cast(counter, tf.float32) / FLAGS.num_steps
        else:
            anneal_const = 1

        if FLAGS.debug:
            cum_energies = tf.Print(cum_energies, [tf.reduce_mean(cum_energies) / (FLAGS.plan_steps - FLAGS.total_frame + 3)])

        # cum_energies = cum_energies + 0.000001 * tf.square(x_joint - X_END)
        anneal_const = tf.cast(counter, tf.float32) / FLAGS.num_steps

        if FLAGS.constraint_vel:
            v = 0.01 * tf.reduce_sum(tf.square((x_joint[:, 1:] - x_joint[:, :-1])))
            cum_energies += v

        if FLAGS.constraint_goal:
            d = tf.reduce_sum(tf.square(x_joint - X_END))
            cum_energies += d

        # TODO change to be the appropriate weight for distance to goal
        cum_energies = cum_energies + 1e-3 * tf.reduce_sum(tf.abs(x_joint[:, -1:] - X_END))

        x_grad, action_grad = tf.gradients(cum_energies, [x_joint, actions])
        x_joint = x_joint - FLAGS.step_lr * anneal_const * x_grad
        x_joint = tf.concat([X_START, x_joint[:, 1:FLAGS.plan_steps + 1]], axis=1)
        x_joint = tf.clip_by_value(x_joint, -1.0, 1.0)

        actions = actions - FLAGS.step_lr * anneal_const * action_grad
        actions = tf.clip_by_value(actions, -1.0, 1.0)

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


def main():
    logdir = osp.join(FLAGS.logdir, FLAGS.exp)
    if not osp.exists(logdir):
        os.makedirs(logdir)
    logger = TensorBoardOutputFormat(logdir)

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

    if FLAGS.cond:
        target_vars = construct_cond_plan_model(model, weights, X_PLAN, X_START, X_END, ACTION_PLAN)
    else:
        target_vars = construct_no_cond_plan_model(model, weights, X_PLAN, X_START, X_END, ACTION_LABEL)

    sess = tf.InteractiveSession()
    saver = loader = tf.train.Saver(max_to_keep=10, keep_checkpoint_every_n_hours=2)

    tf.global_variables_initializer().run()
    print("Initializing variables...")

    if FLAGS.resume_iter != -1:
        model_file = osp.join(logdir, 'model_{}'.format(FLAGS.resume_iter))
        saver.restore(sess, model_file)

    start_arr = [FLAGS.start1, FLAGS.start2]
    end_arr = [FLAGS.end1, FLAGS.end2]

    if FLAGS.datasource == 'point':
        env = Point(start_arr, end_arr, FLAGS.eps, FLAGS.obstacle)
    elif FLAGS.datasource == 'maze':
        # env = Maze([0.1, 0.0], [0.7, -0.8], FLAGS.eps, FLAGS.obstacle)
        env = Maze([-0.85, -0.85], [0.7, -0.8], FLAGS.eps, FLAGS.obstacle)
    else:
        raise KeyError

    get_avg_step_num(target_vars, sess, env)


if __name__ == "__main__":
    main()
