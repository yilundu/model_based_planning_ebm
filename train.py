import datetime
import os
import os.path as osp
import random

import gym
import imageio
import matplotlib as mpl
import matplotlib.patches as patches
import tensorflow as tf
from baselines.logger import TensorBoardOutputFormat
from tensorflow.python.platform import flags

from traj_model import TrajFFDynamics, TrajInverseDynamics, TrajNetLatentFC

mpl.use('Agg')
import matplotlib.pyplot as plt
from tensorflow.core.util import event_pb2
import torch
import numpy as np
from itertools import product
from custom_adam import AdamOptimizer
# from render_utils import render_reach
from utils import ReplayBuffer
import seaborn as sns

sns.set()
plt.rcParams["font.family"] = "Times New Roman"

# from inception import get_inception_score
# from fid import get_fid_score

torch.manual_seed(1)
FLAGS = flags.FLAGS

# Dataset Options
flags.DEFINE_string('type', 'random', 'random or past for initialization of new frame')
flags.DEFINE_string('datasource', 'point', 'point or maze or reacher')
flags.DEFINE_integer('batch_size', 256, 'Size of inputs')
flags.DEFINE_bool('single', False, 'whether to train on a single task')
flags.DEFINE_integer('data_workers', 6, 'Number of different data workers to load data in parallel')

# General Experiment Seittings
flags.DEFINE_string('logdir', 'cachedir', 'location where log of experiments will be stored')
flags.DEFINE_string('imgdir', 'rollout_images', 'location where image results of experiments will be stored')
flags.DEFINE_string('exp', 'default', 'name of experiments')
flags.DEFINE_integer('log_interval', 10, 'log outputs every so many batches')
flags.DEFINE_integer('save_interval', 1000, 'save outputs every so many batches')
flags.DEFINE_integer('test_interval', 1000, 'evaluate outputs every so many batches')
flags.DEFINE_integer('resume_iter', -1, 'iteration to resume training from')
flags.DEFINE_bool('train', True, 'whether to train or test')
flags.DEFINE_bool('debug', False, 'debug what is going on for conditional models')
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
flags.DEFINE_bool('inverse_dynamics', True, 'Whether to train a inverse dynamics model')

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
flags.DEFINE_bool('constraint_vel', False, 'A distance constraint between each subsequent state')
flags.DEFINE_bool('anneal', False, 'Whether to use simulated annealing for sampling')

# use FF to train forward prediction rather than EBM
flags.DEFINE_bool('ff_model', False, 'Run action conditional with a deterministic FF network')

flags.DEFINE_integer('n_exp', 1, 'Number of tests run')

FLAGS.batch_size *= FLAGS.num_gpus

# set_seed(FLAGS.seed)

if FLAGS.datasource == 'point':
    FLAGS.latent_dim = 2
    FLAGS.action_dim = 2
elif FLAGS.datasource == 'maze':
    FLAGS.latent_dim = 2
    FLAGS.action_dim = 2
elif FLAGS.datasource == "reacher":
    FLAGS.latent_dim = 4
    FLAGS.action_dim = 2


def make_image(tensor):
    """Convert an numpy representation image to Image protobuf"""
    from PIL import Image
    if len(tensor.shape) == 4:
        _, height, width, channel = tensor.shape
    elif len(tensor.shape) == 3:
        height, width, channel = tensor.shape
    elif len(tensor.shape) == 2:
        height, width = tensor.shape
        channel = 1
    image = Image.fromarray(tensor)
    import io
    output = io.BytesIO()
    image.save(output, format='PNG')
    image_string = output.getvalue()
    output.close()
    return tf.Summary.Image(height=height,
                            width=width,
                            colorspace=channel,
                            encoded_image_string=image_string)


def compute_lr(itr):
    frac = min((itr + 100) / 300, 1)
    return frac * FLAGS.lr


def log_image(im, logger, tag, step=0):
    im = make_image(im)

    summary = [tf.Summary.Value(tag=tag, image=im)]
    summary = tf.Summary(value=summary)
    event = event_pb2.Event(summary=summary)
    event.step = step
    logger.writer.WriteEvent(event)
    logger.writer.Flush()


def rescale_im(image):
    image = np.clip(image, 0, 1)
    return (image * 255).astype(np.uint8)


def train(target_vars, saver, sess, logger, dataloader, actions, resume_iter):
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
    random_combo = list(product(range(FLAGS.total_frame, dataloader.shape[1] - FLAGS.total_frame),
                                range(0, dataloader.shape[0] - FLAGS.batch_size, FLAGS.batch_size)))

    replay_buffer = ReplayBuffer(10000)

    for epoch in range(FLAGS.epoch_num):
        random.shuffle(random_combo)
        perm_idx = np.random.permutation(dataloader.shape[0])
        for j, i in random_combo:
            label = dataloader[:, j - FLAGS.total_frame:j]
            label_i = label[perm_idx[i:i + FLAGS.batch_size]]
            data_corrupt = np.random.uniform(-1.2, 1.2, (
                FLAGS.batch_size, FLAGS.total_frame, FLAGS.input_objects, FLAGS.latent_dim))

            feed_dict = {X: label_i, X_NOISE: data_corrupt, lr: FLAGS.lr}

            feed_dict[ACTION_LABEL] = actions[perm_idx[i:i + FLAGS.batch_size], j - FLAGS.total_frame + 1]
            feed_dict[ACTION_NOISE_LABEL] = np.random.uniform(-1.2, 1.2, (FLAGS.batch_size, 2))

            # print("Action label", feed_dict[ACTION_LABEL][0])
            # print("Action noise label", feed_dict[ACTION_NOISE_LABEL][0])
            # print(label_i.shape)
            # print("X label", (label_i[:, 1, 0] - label_i[:, 0, 0]) - feed_dict[ACTION_LABEL] / 20)
            # assert False

            if len(replay_buffer) > FLAGS.batch_size:
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


def test(target_vars, saver, sess, logdir, data, actions, dataset_train, mean, std):
    X_START = target_vars['X_START']
    X_END = target_vars['X_END']
    X_PLAN = target_vars['X_PLAN']
    x_joint = target_vars['x_joint']

    n = FLAGS.n_exp

    if FLAGS.datasource == "point":
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

    # interp_weights = np.linspace(0, 1, FLAGS.plan_steps+2)[None, :, None, None]
    # x_plan = interp_weights * x_start + (1 - interp_weights) * x_end
    # x_plan = x_plan[:, 1:-1]

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


def construct_no_cond_plan_model(model, weights, X_PLAN, X_START, X_END, ACTION_LABEL):
    x_joint = tf.concat([X_START, X_PLAN, X_END], axis=1)
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

            if FLAGS.anneal:
                anneal_val = tf.cast(counter, tf.float32) / FLAGS.num_steps
            else:
                anneal_val = 1

            cum_energies = tf.reduce_sum(tf.concat(cum_energies, axis=1), axis=1)

            if FLAGS.constraint_vel:
                cum_energies = cum_energies + 0.0001 * tf.reduce_sum(tf.square(x_joint[:, 1:] - x_joint[:, :-1]))

            x_grad = tf.gradients(cum_energies, [x_joint])[0]
            x_joint = x_joint - FLAGS.step_lr * anneal_val * x_grad

        # Reset the start and end states to be previous values
        x_joint = tf.concat([X_START, x_joint[:, 1:FLAGS.plan_steps + 1], X_END], axis=1)
        counter = counter + 1

        counter = tf.Print(counter,
                           [tf.reduce_mean(cum_energies), tf.reduce_max(cum_energies), tf.reduce_min(cum_energies)])

        if FLAGS.datasource == "maze" or FLAGS.datasource == "point":
            x_joint = tf.clip_by_value(x_joint, -1.0, 1.0)

        return counter, x_joint

    steps, x_joint = tf.while_loop(c, mcmc_step, (steps, x_joint))
    target_vars = {}
    target_vars['x_joint'] = x_joint
    target_vars['X_START'] = X_START
    target_vars['X_END'] = X_END
    target_vars['ACTION_LABEL'] = ACTION_LABEL
    target_vars['X_PLAN'] = X_PLAN

    return target_vars


def construct_cond_plan_model(model, weights, X_PLAN, X_START, X_END, ACTION_PLAN, ff=False):
    actions = ACTION_PLAN
    x_joint = tf.concat([X_START, X_PLAN, X_END], axis=1)
    steps = tf.constant(0)
    c = lambda i, x, y: tf.less(i, FLAGS.num_steps)

    if ff:
        FLAGS.total_frame = 1

    def mcmc_step(counter, x_joint, actions):
        actions = actions + tf.random_normal(tf.shape(actions), mean=0.0, stddev=0.01)
        x_joint = x_joint + tf.random_normal(tf.shape(x_joint), mean=0.0, stddev=0.01)
        cum_energies = 0
        for i in range(FLAGS.plan_steps - FLAGS.total_frame + 3):
            cum_energy = model.forward(x_joint[:, i:i + FLAGS.total_frame], weights, action_label=actions[:, i])
            cum_energies = cum_energies + cum_energy

        cum_energies = tf.Print(cum_energies, [cum_energies], message="energies")

        if FLAGS.anneal:
            anneal_val = tf.cast(counter, tf.float32) / FLAGS.num_steps
        else:
            anneal_val = 1

        x_grad, action_grad = tf.gradients(cum_energies, [x_joint, actions])
        x_joint = x_joint - FLAGS.step_lr * anneal_val * x_grad
        x_joint = tf.concat([X_START, x_joint[:, 1:FLAGS.plan_steps + 1], X_END], axis=1)
        x_joint = tf.clip_by_value(x_joint, -1.0, 1.0)

        actions = actions - FLAGS.step_lr * anneal_val * action_grad
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


def construct_model(model, weights, X_NOISE, X, ACTION_LABEL, ACTION_NOISE_LABEL, LR, optimizer):
    target_vars = {}
    x_mods = []

    energy_pos = model.forward(X, weights, action_label=ACTION_LABEL)
    energy_noise = model.forward(X_NOISE, weights, reuse=True, stop_at_grad=True, action_label=ACTION_LABEL)

    print("Building graph...")
    x_mod = X_NOISE

    x_grads = []
    x_ees = []
    energy_negs = [energy_noise]
    loss_energys = []

    if FLAGS.inverse_dynamics:
        dyn_model = TrajInverseDynamics(dim_input=FLAGS.latent_dim, dim_output=FLAGS.action_dim)
        weights = dyn_model.construct_weights(scope="inverse_dynamics", weights=weights)

    if FLAGS.ff_model:
        ff_model = TrajFFDynamics(dim_input=FLAGS.latent_dim, dim_output=FLAGS.latent_dim)
        weights = ff_model.construct_weights(scope="ff_model", weights=weights)

    steps = tf.constant(0)
    c = lambda i, x, y: tf.less(i, FLAGS.num_steps)

    def mcmc_step(counter, x_mod, action_label):
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
            action_label = action_label + tf.random_normal(tf.shape(action_label), mean=0.0, stddev=0.01)

            energy_noise = model.forward(x_mod, weights, action_label=action_label, reuse=True, stop_at_grad=True)
            lr = FLAGS.step_lr

            x_grad = tf.gradients(FLAGS.temperature * energy_noise, [x_mod])[0]

            x_mod = x_mod - lr * x_grad

            if FLAGS.cond:
                x_grad, action_grad = tf.gradients(FLAGS.temperature * energy_noise, [x_mod, action_label])
            else:
                x_grad, action_grad = tf.gradients(FLAGS.temperature * energy_noise, [x_mod])[0], tf.zeros(1)

            action_label = action_label - FLAGS.step_lr * action_grad

        # x_mod = tf.clip_by_value(x_mod, -1.2, 1.2)
        # action_label = tf.clip_by_value(action_label, -1.2, 1.2)

        counter = counter + 1

        return counter, x_mod, action_label

    print(ACTION_NOISE_LABEL.get_shape(), ACTION_LABEL.get_shape())
    steps, x_mod, action_label = tf.while_loop(c, mcmc_step, (steps, x_mod, ACTION_NOISE_LABEL))
    # action_label = tf.Print(action_label, [action_label], "action label (HELP ME)")

    if FLAGS.cond:
        if FLAGS.datasource != "reacher":
            progress_diff = tf.reduce_mean(tf.abs((x_mod[:, 1, 0] - x_mod[:, 0, 0]) - action_label / 20))
        else:
            progress_diff = tf.zeros(1)
        # progress_diff = tf.reduce_mean(tf.abs((X[:, 1, 0] - X[:, 0, 0]) - ACTION_LABEL / 20))
    else:
        progress_diff = tf.zeros(1)

    target_vars['x_mod'] = x_mod
    temp = FLAGS.temperature

    loss_energy = temp * model.forward(x_mod, weights, reuse=True, action_label=action_label, stop_grad=True)
    x_mod = tf.stop_gradient(x_mod)
    action_label = tf.stop_gradient(action_label)

    energy_neg = model.forward(x_mod, weights, action_label=action_label, reuse=True)
    if FLAGS.cond:
        x_grad, action_grad = tf.gradients(FLAGS.temperature * energy_neg, [x_mod, action_label])
    else:
        x_grad, action_grad = tf.gradients(FLAGS.temperature * energy_neg, [x_mod])[0], tf.zeros(1)
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

    if FLAGS.inverse_dynamics:
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
        if FLAGS.inverse_dynamics:
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

    if FLAGS.datasource == 'point' or FLAGS.datasource == 'maze' or FLAGS.datasource == 'reacher':
        model = TrajNetLatentFC(dim_input=FLAGS.latent_dim)
        X_NOISE = tf.placeholder(shape=(None, FLAGS.total_frame, FLAGS.input_objects, FLAGS.latent_dim),
                                 dtype=tf.float32)
        X = tf.placeholder(shape=(None, FLAGS.total_frame, FLAGS.input_objects, FLAGS.latent_dim), dtype=tf.float32)

        ACTION_LABEL = tf.placeholder(shape=(None, 2), dtype=tf.float32)
        ACTION_NOISE_LABEL = tf.placeholder(shape=(None, 2), dtype=tf.float32)
        ACTION_PLAN = tf.placeholder(shape=(None, FLAGS.plan_steps + 1, 2), dtype=tf.float32)

        X_START = tf.placeholder(shape=(None, 1, FLAGS.input_objects, FLAGS.latent_dim), dtype=tf.float32)
        X_PLAN = tf.placeholder(shape=(None, FLAGS.plan_steps, FLAGS.input_objects, FLAGS.latent_dim), dtype=tf.float32)
        X_END = tf.placeholder(shape=(None, 1, FLAGS.input_objects, FLAGS.latent_dim), dtype=tf.float32)
    else:
        raise AssertionError("Unsupported data source")

    weights = model.construct_weights(action_size=FLAGS.action_dim)
    LR = tf.placeholder(tf.float32, [])
    optimizer = AdamOptimizer(LR, beta1=0.0, beta2=0.999)

    if FLAGS.train or FLAGS.debug:
        target_vars = construct_model(model, weights, X_NOISE, X, ACTION_LABEL, ACTION_NOISE_LABEL, LR, optimizer)
    else:
        # evaluation
        if FLAGS.cond:
            target_vars = construct_cond_plan_model(model, weights, X_PLAN, X_START, X_END, ACTION_PLAN, FLAGS.ff_model)
        else:
            target_vars = construct_no_cond_plan_model(model, weights, X_PLAN, X_START, X_END, ACTION_LABEL)

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

    if FLAGS.datasource == 'point':
        dataset = np.load('point.npz')['obs'][:, :, None, :]
        actions = np.load('point.npz')['action']
        mean, std = 0, 1
    elif FLAGS.datasource == 'maze':
        dataset = np.load('maze.npz')['obs'][:, :, None, :]
        actions = np.load('maze.npz')['action']
        mean, std = 0, 1
    elif FLAGS.datasource == "reacher":
        dataset = np.load('reacher.npz')['obs'][:, :, None, :]
        actions = np.load('reacher.npz')['action']
        dones = np.load('reacher.npz')['action']

        dataset[:, :, :, :2] = dataset[:, :, :, :2] % (2 * np.pi)
        s = dataset.shape

        dataset_flat = dataset.reshape((-1, FLAGS.latent_dim))
        # dataset = dataset / 55.
        mean, std = dataset_flat.mean(axis=0), dataset_flat.std(axis=0)
        std = std + 1e-5
        dataset = (dataset - mean) / std
        print(dataset.max(), dataset.min())

        # For now a hacky way to deal with dones since each episode is always of length 50
        dataset = np.concatenate([dataset[:, 49:99], dataset[:, [99] + list(range(49))]], axis=0)
        actions = np.concatenate([actions[:, 49:99], actions[:, [99] + list(range(49))]], axis=0)
    else:
        raise AssertionError("Unsupported data source")

    if FLAGS.single:
        dataset = np.tile(dataset[0:1], (100, 1, 1, 1))[:, :20]

    split_idx = int(dataset.shape[0] * 0.9)

    dataset_train = dataset[:split_idx]
    actions_train = actions[:split_idx]
    dataset_test = dataset[split_idx:]
    actions_test = actions[split_idx:]

    if FLAGS.train:
        train(target_vars, saver, sess, logger, dataset_train, actions_train, resume_itr)

    if FLAGS.debug:
        debug(target_vars, sess)
    else:
        test(target_vars, saver, sess, logdir, dataset_test, actions_test, dataset_train, mean, std)


if __name__ == "__main__":
    main()
