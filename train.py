import tensorflow as tf
import numpy as np
from tensorflow.python.platform import flags
from traj_model import TrajNetLatent, TrajNetLatentGen, TrajNetLatentFC, TrajNetLatentGenFC
import os.path as osp
import os
from rl_algs.logger import TensorBoardOutputFormat
from utils import average_gradients, set_seed
from tqdm import tqdm
import random
import time as time
from io import StringIO
import matplotlib.pyplot as plt
from tensorflow.core.util import event_pb2
import torch
import numpy as np
import imageio as io
from itertools import product
import random
from custom_adam import AdamOptimizer
from collections import defaultdict
from render_utils import render_reach
from utils import ReplayBuffer, calculate_frechet_distance

# from inception import get_inception_score
# from fid import get_fid_score

torch.manual_seed(1)
FLAGS = flags.FLAGS

# Dataset Options
flags.DEFINE_string('type', 'past', 'random or past for initialization of new frame')
flags.DEFINE_string('datasource', 'fetch', 'traj or fetch or hand')
flags.DEFINE_bool('image', False, 'wheter to train with images or 2d trajectories')
flags.DEFINE_integer('batch_size', 1024, 'Size of inputs')
flags.DEFINE_bool('single', False, 'whether to train on a single task')
flags.DEFINE_integer('data_workers', 6, 'Number of different data workers to load data in parallel')
flags.DEFINE_integer('im_size', 32, 'Size of images to train')

# General Experiment Seittings
flags.DEFINE_string('logdir', 'cachedir', 'location where log of experiments will be stored')
flags.DEFINE_string('exp', 'default', 'name of experiments')
flags.DEFINE_integer('log_interval', 10, 'log outputs every so many batches')
flags.DEFINE_integer('save_interval', 1000, 'save outputs every so many batches')
flags.DEFINE_integer('test_interval', 1000, 'evaluate outputs every so many batches')
flags.DEFINE_integer('resume_iter', -1, 'iteration to resume training from')
flags.DEFINE_bool('train', True, 'whether to train or test')
flags.DEFINE_integer('epoch_num', 10, 'Number of Epochs to train on')
flags.DEFINE_float('lr', 1e-3, 'Learning for training')
flags.DEFINE_integer('seed', 0, 'Value of seed')

# Custom Experiments Settings
flags.DEFINE_integer('num_gpus', 1, 'number of gpus to train on')
flags.DEFINE_bool('gp', False, 'whether to train with gradient penalty')
flags.DEFINE_bool('ee', False, 'whether to train with expert iteration(using future gradient descent for energies)')
flags.DEFINE_bool('cd', True, 'whether to train with expert iteration(using future gradient descent for energies)')
flags.DEFINE_float('gp_coeff', 10.0, 'Coefficient to multiply the gradient penalty')
flags.DEFINE_float('ee_coeff', 1.0, 'Coefficient to multiply the expert iteration')
flags.DEFINE_float('ml_coeff', 1.0, 'Coefficient to multiply maximum likelihood (descriminator coefficient)')
flags.DEFINE_float('reg_scale', 1.0, 'Scale of regularization')

flags.DEFINE_integer('temperature', 1, 'Temperature for energy function')
flags.DEFINE_integer('num_steps', 20, 'Steps of gradient descent for training')
flags.DEFINE_integer('eval_steps', 20, 'Steps of gradient descent for evaluation')

# For past init
# flags.DEFINE_float('step_lr', 0.01, 'Size of steps for gradient descent')
# For random init
flags.DEFINE_float('step_lr', 1.0, 'Size of steps for gradient descent')

flags.DEFINE_bool('grad_free', False, 'instead of using gradient descent to generate latents, use DFO')
flags.DEFINE_integer('noise_sim', 20, 'Number of forward evolution steps to calculate')
flags.DEFINE_bool('supervised', False, 'instead of using unsupervised criterion use l2 distance')
flags.DEFINE_bool('gen_network', False, 'instead of using gradient descent, just use a generation network')
flags.DEFINE_bool('logsumexp', True, 'Use the logsumexp criterion instead of sigmoid for training model')

# Projected gradient descent
flags.DEFINE_float('proj_norm', 0.00, 'Maximum change of input images')
flags.DEFINE_string('proj_norm_type', 'li', 'What type of ball for projection, only support l2 and li')

# Architecture Settings
flags.DEFINE_bool('max_pool', False, 'Whether or not to use max pooling rather than strided convolutions')
flags.DEFINE_integer('num_filters', 64, 'number of filters for conv nets -- 32 for miniimagenet, 64 for omniglot.')
flags.DEFINE_bool('bn', False, 'Whether to use batch normalization or not')
flags.DEFINE_bool('spec_norm', False, 'Whether to use spectral normalization on weights')
flags.DEFINE_bool('use_bias', True, 'Whether to use bias in convolution')
flags.DEFINE_integer('input_objects', 1, 'Number of objects to predict the trajectory of.')
flags.DEFINE_integer('latent_dim', 24, 'Number of dimension encoding state of object')
flags.DEFINE_integer('action_dim', 20, 'Number of dimension encoding action of object')
flags.DEFINE_bool('second_order', False, 'Whether to use second order methods to generate solutions')
flags.DEFINE_bool('second_seq_opt', False, 'Use second order to generate solutions of sequence of pairs')

# Physics options
flags.DEFINE_bool('test_gt', False, 'Whether or not to use test_gt')
flags.DEFINE_bool('train_gt', True, 'Whether or not to train using ground truth frames')
flags.DEFINE_integer('train_pred_steps', 1, 'Number of steps to train on using predicted frames')
flags.DEFINE_bool('spatial_conv', True, 'Use spatial convolutions')
flags.DEFINE_integer('label_frame', 4, 'Number of label frames')
flags.DEFINE_integer('pred_frame', 1, 'Number of predicted frames')
flags.DEFINE_bool('seq_update', False, 'Whether to do sequential updates to predicted value')

flags.DEFINE_bool('replay_batch', False, 'Whether to use a replay buffer for samples')
flags.DEFINE_bool('no_cond', False, 'Whether to condition on actions')
flags.DEFINE_bool('quick', False, 'For quick evaluation of stuff')
flags.DEFINE_bool('zero_kl', False, 'whether to make the kl be zero')

FLAGS.batch_size *= FLAGS.num_gpus

set_seed(FLAGS.seed)

if FLAGS.datasource == 'fetch':
    FLAGS.latent_dim = 24
    FLAGS.action_dim = 20

elif FLAGS.datasource == 'hand':
    FLAGS.latent_dim = 38
    FLAGS.action_dim = 20

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


def frechet_score(pred_flat, data_flat):

    f_scores = []
    for i in range(pred_flat.shape[1]):
        u1, u2 = np.mean(pred_flat[:, i], axis=0), np.mean(data_flat[:, i], axis=0)
        c1, c2 = np.cov(pred_flat[:, i], rowvar=0), np.cov(data_flat[:, i], rowvar=0)

        f_score = calculate_frechet_distance(u1, c1, u2, c2)
        f_scores.append(f_score)

    print("Overall frechet score of {}".format(np.mean(f_scores)))
    return f_scores


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
    x_off = target_vars['x_off']
    temp = target_vars['temp']
    loss_ee = target_vars['loss_ee']
    loss_gp = target_vars['loss_gp']
    x_mod = target_vars['x_mod']
    LABEL = target_vars['LABEL']
    weights = target_vars['weights']
    test_x_mod = target_vars['test_x_mod']
    lr = target_vars['lr']
    x_mods = target_vars['x_mods']
    ACTION_LABEL = target_vars['ACTION_LABEL']

    val_output = [test_x_mod]

    gvs_dict = dict(gvs)

    # remove gradient logging since it is slow
    log_output = [train_op, energy_pos, energy_neg, loss_energy, loss_ml, loss_ee, loss_total, x_grad, x_off, x_mod, *gvs_dict.keys()]
    output = [train_op, x_mod]
    # output.extend(x_mods)

    itr = resume_iter
    x_mod = None
    gd_steps = 1

    train_pred_steps = FLAGS.train_pred_steps

    if FLAGS.train_gt:
        train_pred_steps = 0

    random_combo = list(product(range(FLAGS.label_frame, dataloader.shape[1]-train_pred_steps-FLAGS.pred_frame), range(0, dataloader.shape[0]-FLAGS.batch_size, FLAGS.batch_size)))

    replay_buffer = ReplayBuffer(10000)

    for epoch in range(FLAGS.epoch_num):
        random.shuffle(random_combo)
        perm_idx = np.random.permutation(dataloader.shape[0])
        for j, i in random_combo:
            label = dataloader[:, j-FLAGS.label_frame:j]
            for k in range(train_pred_steps+1):
                data_i = dataloader[perm_idx[i:i+FLAGS.batch_size], j+k:j+k+FLAGS.pred_frame]
                if k == 0:
                    label_i = label[perm_idx[i:i+FLAGS.batch_size]]
                else:
                    label_i = np.concatenate([label_i[:, 1:, :, :], x_mod[:, 0:1, :, :]], axis=1)

                if FLAGS.type == 'random':
                    data_corrupt = np.random.uniform(-1.0, 1.0, (FLAGS.batch_size, FLAGS.pred_frame, FLAGS.input_objects, FLAGS.latent_dim))
                elif FLAGS.type == 'past':
                    data_corrupt = np.tile(label_i[:, -1:], (1, FLAGS.pred_frame, 1, 1))
                    data_corrupt += np.random.uniform(-0.3, 0.3, size=data_corrupt.shape)

                if FLAGS.replay_batch and (x_mod is not None):
                    replay_buffer.add(x_mod)

                    if len(replay_buffer) > FLAGS.batch_size:
                        replay_batch = replay_buffer.sample(FLAGS.batch_size)
                        replay_mask = (np.random.uniform(0, 1, (FLAGS.batch_size)) > 0.1)
                        replay_batch = np.clip(replay_batch, -1, 1)
                        data_corrupt[replay_mask] = replay_batch[replay_mask]

                for l in range(gd_steps):
                    feed_dict = {X_NOISE: data_corrupt, X: data_i, LABEL: label_i, ACTION_LABEL: actions[perm_idx[i:i+FLAGS.batch_size], j]}
                    feed_dict[lr] = compute_lr(itr)

                    if itr % FLAGS.log_interval == 0:
                        _, e_pos, e_neg, loss_e, loss_ml, loss_ee, loss_total, x_grad, x_off, x_mod, *grads = sess.run(log_output, feed_dict)

                        kvs = {}
                        kvs['e_pos'] = e_pos.mean()
                        kvs['temp'] = temp
                        kvs['e_neg'] = e_neg.mean()
                        kvs['loss_e'] = loss_e.mean()

                        kvs['loss_ml'] = loss_ml.mean()
                        kvs['loss_total'] = loss_total.mean()
                        kvs['x_grad'] = np.abs(x_grad).mean()
                        kvs['x_off'] = x_off.mean()
                        kvs['iter'] = itr

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

                    if itr % FLAGS.save_interval == 0:
                        saver.save(sess, osp.join(FLAGS.logdir, FLAGS.exp, 'model_{}'.format(itr)))

                itr += 1

    saver.save(sess, osp.join(FLAGS.logdir, FLAGS.exp, 'model_{}'.format(itr)))

def test(target_vars, saver, sess, logdir, data, actions, mean, std, dataset_train):
    X_NOISE = target_vars['X_NOISE']
    X = target_vars['X']
    LABEL = target_vars['LABEL']
    energy_start = target_vars['energy_start']
    energy_neg = target_vars['energy_neg']
    energy_pos = target_vars['energy_pos']
    x_mod = target_vars['test_x_mod']
    x_joint_mod = target_vars['x_joint_mod']
    label_joint_mod = target_vars['label_joint_mod']
    ACTION_LABEL = target_vars['ACTION_LABEL']

    np.random.seed(1)
    # random.seed(1)

    output = [x_mod]
    joint_output = [x_joint_mod, label_joint_mod]

    data_full = data
    actions_full = actions

    data = data[:20]
    actions = actions[:20]

    prev_context = data[:, :FLAGS.label_frame]

    im_list = []

    train_flat = dataset_train.reshape(dataset_train.shape[0], dataset_train.shape[1], -1)
    test_flat = data.reshape(data.shape[0], data.shape[1], -1)
    frechet_score(train_flat, test_flat)

    # Generate a heatmap of the energy of different coordinate of trajectories
    # creates a heat map of the relative energies of solutions to ball trajectories
    ndim = 32

    # def grid_search(n):
    #     return -2 + n * (4 / (ndim - 1))

    # def reverse_grid_search(n):
    #     return (n+2) * (ndim - 1) / 4

    # heatmap = np.zeros((ndim, ndim))
    # for i in range(ndim):
    #     for j in range(ndim):
    #         x = np.array([grid_search(i), grid_search(j)]).reshape((1, 1, 1, 2))
    #         label = data[:1, :FLAGS.label_frame, :, :]
    #         e_pos = sess.run([energy_pos], {LABEL:label , X:x})
    #         heatmap[i, j] = FLAGS.temperature * e_pos[0]

    # construct_heatmap(heatmap)
    # gt = data[0, FLAGS.label_frame, 0]
    # x, y = reverse_grid_search(gt[0]), reverse_grid_search(gt[1])
    # plt.plot(x, y, 'bo')
    # plt.savefig(osp.join(logdir, "heatmap.png"))


    # Generate and optimize trajectories
    np.random.seed(7)
    output_latent = data[:, :FLAGS.label_frame]

    print(data[0, 0, 0])
    for i in range(FLAGS.label_frame, data.shape[1], FLAGS.pred_frame):
        prev_context = output_latent[:, -FLAGS.label_frame:]
        # prev_context = data[:, i-FLAGS.label_frame:i]

        if FLAGS.type == 'random':
            init_info = np.random.uniform(-0.5, 0.5, (20, FLAGS.pred_frame, FLAGS.input_objects, FLAGS.latent_dim))
        elif FLAGS.type == 'past':
            init_info = np.tile(prev_context[:, -1:], (1, FLAGS.pred_frame, 1, 1))
            init_info += np.random.uniform(-0.3, 0.3, size=init_info.shape)

        output_info = sess.run(output, {X_NOISE: init_info, LABEL: prev_context, ACTION_LABEL: actions[:, i]})[0]
        output_latent = np.concatenate([output_latent, output_info], axis=1)


    # for _ in range(10):
    #     for i in range(FLAGS.label_frame, output_latent.shape[1]-5):
    #         label = output_latent[:, i:i+4]
    #         init_info = output_latent[:, i+4:i+5]
    #         action_label = #
    #         x_joint_mod, label_joint_mod = sess.run(joint_output, {X_NOISE: init_info, LABEL: label, ACTION_LABEL: action_label})
    #         output_latent[:, i:i+4] = label_joint_mod
    #         output_latent[:, i+4:i+5] = x_joint_mod

    # Use this step to generate images

    im_size = 200
    for j in range(0, output_latent.shape[1]):
        panel_frame = np.zeros((20, im_size, 2*im_size, 3))
        label_info = data[:, j]
        output_info = output_latent[:, j]
        label_frame = render_reach(label_info, FLAGS.datasource, mean, std, im_size=im_size)
        output_frame = render_reach(output_info, FLAGS.datasource, mean, std, im_size=im_size)

        panel_frame[:, :, :im_size, :] = label_frame
        panel_frame[:, :, im_size:, :] = output_frame

        im_list.append(panel_frame)

    im_list = np.stack(im_list, axis=1).astype(np.uint8)

    for i in range(20):
        io.mimsave(osp.join(logdir, "test_{}.gif".format(i)), list(im_list[i]))

    return
    # assert (FLAGS.pred_frame == 1)
    # assert (FLAGS.label_frame == 4)

    n = 50
    mse_map = defaultdict(list)
    # Compute MSE results for up to 10 steps in the future
    for i in range(0, data_full.shape[0]-FLAGS.batch_size, FLAGS.batch_size):
        data = data_full[i: i+FLAGS.batch_size]
        actions = actions_full[i:i+FLAGS.batch_size]
        for j in tqdm(range(FLAGS.label_frame, data_full.shape[1] - n)):
            prev_context = data[:, j-FLAGS.label_frame:j]
            total_output = data[:, j-FLAGS.label_frame:j]
            for k in range(n):
                if FLAGS.type == 'random':
                    init_info = np.random.uniform(-0.5, 0.5, (FLAGS.batch_size, FLAGS.pred_frame, FLAGS.input_objects, FLAGS.latent_dim))
                elif FLAGS.type == 'past':
                    init_info = np.tile(prev_context[:, -1:], (1, FLAGS.pred_frame, 1, 1))
                    init_info += np.random.uniform(-0.3, 0.3, size=init_info.shape)

                output_info = sess.run(output, {X_NOISE: init_info, LABEL: prev_context, ACTION_LABEL: actions[:, j+k]})[0]
                output_latent = np.concatenate([prev_context, output_info], axis=1)
                total_output = np.concatenate([total_output, output_info], axis=1)
                prev_context = output_latent[:, 1:]

            for _ in range(0):
                for i in range(4, total_output.shape[1]-5):
                    label = total_output[:, i:i+4]
                    init_info = total_output[:, i+4:i+5]
                    action_label = actions[:, j+i+4-FLAGS.label_frame]

                    if i >= 4:
                        x_joint_mod, label_joint_mod = sess.run(joint_output, {X_NOISE: init_info, LABEL: label, ACTION_LABEL: action_label})
                        total_output[:, i:i+4] = label_joint_mod
                        total_output[:, i+4:i+5] = x_joint_mod
                    else:
                        output_info = sess.run(output, {X_NOISE: init_info, LABEL: label, ACTION_LABEL: action_label})
                        total_output[:, i+4:i+5] = output_info



            for k in range(n):
                output_gt = data[:, j+k:j+k+FLAGS.pred_frame]
                output_info= total_output[:, 4+k:4+k+FLAGS.pred_frame]
                mse_error = np.square(output_gt - output_info).mean()
                mse_map[k].append(mse_error)

            if FLAGS.quick:
                break


    for i in range(n):
        print("Error of {} for {} step predictions".format(np.array(mse_map[i]).mean(), i))


    # Fit Gaussians to each point and generate the Frechet distance between Gaussian between 
    output_latents = []

    n = data_full.shape[0] // FLAGS.batch_size
    pred_trajs = []
    for data, actions in zip(np.array_split(data_full, n), np.array_split(actions_full, n)):

        output_latent = data[:, :FLAGS.label_frame]
        for i in range(FLAGS.label_frame, data.shape[1], FLAGS.pred_frame):
            prev_context = output_latent[:, -FLAGS.label_frame:]

            if FLAGS.type == 'random':
                init_info = np.random.uniform(-0.5, 0.5, (FLAGS.batch_size, FLAGS.pred_frame, FLAGS.input_objects, FLAGS.latent_dim))
            elif FLAGS.type == 'past':
                init_info = np.tile(prev_context[:, -1:], (1, FLAGS.pred_frame, 1, 1))
                init_info += np.random.uniform(-0.3, 0.3, size=init_info.shape)

            output_info = sess.run(output, {X_NOISE: init_info, LABEL: prev_context, ACTION_LABEL: actions[:, i]})[0]
            output_latent = np.concatenate([output_latent, output_info], axis=1)

        pred_trajs.append(output_latent)

    pred_trajs = np.concatenate(pred_trajs, axis=0)

    pred_flat = pred_trajs.reshape(pred_trajs.shape[0], pred_trajs.shape[1], -1)
    data_flat = data_full.reshape(data_full.shape[0], data_full.shape[1], -1)

    assert (pred_flat.shape == data_flat.shape)

    f_scores = frechet_score(pred_flat, data_flat)
    np.save("{}_frechet_score.npy".format(FLAGS.exp), f_scores)


def main():
    logdir = osp.join(FLAGS.logdir, FLAGS.exp)
    if not osp.exists(logdir):
        os.makedirs(logdir)
    logger = TensorBoardOutputFormat(logdir)

    # Only know the setting for omniglot, not sure about others
    batch_size = FLAGS.batch_size

    if FLAGS.datasource == 'traj':
        model = TrajNetLatent(dim_input=FLAGS.latent_dim)
        X_NOISE = tf.placeholder(shape=(None, FLAGS.pred_frame, FLAGS.input_objects, FLAGS.latent_dim), dtype=tf.float32)
        X = tf.placeholder(shape=(None, FLAGS.pred_frame, FLAGS.input_objects, FLAGS.latent_dim), dtype = tf.float32)
        LABEL = tf.placeholder(shape=(None, FLAGS.label_frame, FLAGS.input_objects, FLAGS.latent_dim), dtype=tf.float32)
        ACTION_LABEL = tf.placeholder(shape=(None, 20), dtype=tf.float32)
    elif FLAGS.datasource == 'fetch':
        model = TrajNetLatent(dim_input=FLAGS.latent_dim)
        X_NOISE = tf.placeholder(shape=(None, FLAGS.pred_frame, FLAGS.input_objects, FLAGS.latent_dim), dtype=tf.float32)
        X = tf.placeholder(shape=(None, FLAGS.pred_frame, FLAGS.input_objects, FLAGS.latent_dim), dtype = tf.float32)
        LABEL = tf.placeholder(shape=(None, FLAGS.label_frame, FLAGS.input_objects, FLAGS.latent_dim), dtype=tf.float32)
        ACTION_LABEL = tf.placeholder(shape=(None, 20), dtype=tf.float32)
    elif FLAGS.datasource == 'hand':
        # model = TrajNetLatent(dim_input=FLAGS.latent_dim)
        model = TrajNetLatentFC(num_filters=128, dim_input=FLAGS.latent_dim)
        X_NOISE = tf.placeholder(shape=(None, FLAGS.pred_frame, FLAGS.input_objects, FLAGS.latent_dim), dtype=tf.float32)
        X = tf.placeholder(shape=(None, FLAGS.pred_frame, FLAGS.input_objects, FLAGS.latent_dim), dtype = tf.float32)
        LABEL = tf.placeholder(shape=(None, FLAGS.label_frame, FLAGS.input_objects, FLAGS.latent_dim), dtype=tf.float32)
        ACTION_LABEL = tf.placeholder(shape=(None, 20), dtype=tf.float32)

    weights = model.construct_weights(action_size=20)


    # Varibles to run in training
    X_SPLIT = tf.split(X, FLAGS.num_gpus)
    X_NOISE_SPLIT = tf.split(X_NOISE, FLAGS.num_gpus)
    LABEL_SPLIT = tf.split(LABEL, FLAGS.num_gpus)
    X_NOISE_JOINT_SPLIT = X_NOISE_SPLIT.copy()
    LABEL_JOINT_SPLIT = LABEL_SPLIT.copy()
    tower_grads = []

    LR = tf.placeholder(tf.float32, [])

    if FLAGS.gen_network:
        gen_model = TrajNetLatentGenFC(dim_input=FLAGS.latent_dim)
        weights_other = gen_model.construct_weights(action_size=20)
        optimizer = AdamOptimizer(LR)
    else:
        optimizer = AdamOptimizer(LR, beta1=0.0, beta2=0.999)

    target_vars = {}
    x_mods = []

    for j in range(FLAGS.num_gpus):
        # with tf.device('/gpu:{}'.format(j)):
            energy_pos = model.forward(X_SPLIT[j], weights, label=LABEL_SPLIT[j], action_label=ACTION_LABEL)
            energy_noise = energy_start = model.forward(X_NOISE_SPLIT[j], weights, label=LABEL_SPLIT[j], reuse=True, stop_at_grad=True, action_label=ACTION_LABEL)

            print("Building graph...")
            x_mod = X_NOISE_SPLIT[j]
            x_joint_mod = X_NOISE_JOINT_SPLIT[j]
            label_joint_mod = LABEL_JOINT_SPLIT[j]

            # ee_step = FLAGS.num_steps // 6
            x_grads = []
            x_ees = []

            energy_negs = [energy_noise]
            loss_energys = []

            for i in range(FLAGS.num_steps):
                if FLAGS.grad_free:
                    if FLAGS.seq_update:
                        for n in range(FLAGS.pred_frame):
                            x_mod_neg = tf.tile(tf.expand_dims(x_mod, axis=1), (1, FLAGS.noise_sim, 1, 1, 1))
                            x_mod_neg_shape = tf.shape(x_mod_neg)

                            # User energies to set movement speed in derivative free optimization
                            energy_noise = model.forward(x_mod, weights, label=LABEL_SPLIT[j], reuse=True, action_label=ACTION_LABEL)
                            energy_noise = tf.reshape(energy_noise, (x_mod_neg_shape[0], 1, 1, 1, 1))

                            x_mod_modify = x_mod_neg[:, :, n:n+1, :, :]
                            x_mod_modify_shape = tf.shape(x_mod_modify)
                            x_noise =  energy_noise * tf.random_normal(x_mod_modify_shape, mean=0.0, stddev=0.01)

                            x_mod_modify = x_mod_modify + x_noise

                            x_mod_stack = x_mod_neg = tf.concat([x_mod_neg[:, :, :n, :, :], x_mod_modify, x_mod_neg[:, :, n+1:, :, :]], axis=2)
                            x_mod_neg = tf.reshape(x_mod_neg, (x_mod_neg_shape[0]*x_mod_neg_shape[1], x_mod_neg_shape[2], x_mod_neg_shape[3], x_mod_neg_shape[4]))

                            # Tile the label tensor
                            label_noise = tf.tile(tf.expand_dims(LABEL_SPLIT[j], axis=1), (1, FLAGS.noise_sim, 1, 1, 1))
                            label_shape = tf.shape(label_noise)
                            label_noise = tf.reshape(label_noise, (label_shape[0]*label_shape[1], label_shape[2], label_shape[3], label_shape[4]))

                            energy_noise = -1 * model.forward(x_mod_neg, weights, label=label_noise, action_label=ACTION_LABEL, reuse=True)
                            energy_noise = tf.reshape(energy_noise, (x_mod_neg_shape[0], FLAGS.noise_sim))
                            energy_wt = tf.nn.softmax(energy_noise, axis=1)
                            energy_wt = tf.reshape(energy_wt, (x_mod_neg_shape[0], FLAGS.noise_sim, 1, 1, 1))
                            x_mod = tf.reduce_sum(energy_wt * x_mod_stack, axis=1)
                            x_grad = tf.zeros(1)
                    else:
                        # Use gradients to sample the distributions
                        x_mod_neg = tf.tile(tf.expand_dims(x_mod, axis=1), (1, FLAGS.noise_sim, 1, 1, 1))
                        x_mod_neg_shape = tf.shape(x_mod_neg)

                        # User energies to set movement speed in derivative free optimization
                        energy_noise = FLAGS.temperature * model.forward(x_mod, weights, label=LABEL_SPLIT[j], action_label=ACTION_LABEL, reuse=True)
                        print(energy_noise.get_shape())
                        # energy_noise = tf.tile(energy_noise, (FLAGS.noise_sim,))
                        energy_noise = tf.reshape(energy_noise, (x_mod_neg_shape[0], 1, 1, 1, 1))

                        # Use stddev 0.7 for particles!!!!
                        x_noise =  tf.random_normal(x_mod_neg_shape, mean=0.0, stddev=0.1)
                        x_mod_stack = x_mod_neg = x_mod_neg + x_noise
                        x_mod_neg = tf.reshape(x_mod_neg, (x_mod_neg_shape[0]*x_mod_neg_shape[1], x_mod_neg_shape[2], x_mod_neg_shape[3], x_mod_neg_shape[4]))
                        # x_mod_neg = tf.Print(x_mod_neg, [x_mod_neg, x_mod])

                        # Tile the label tensor
                        label_noise = tf.tile(tf.expand_dims(LABEL_SPLIT[j], axis=1), (1, FLAGS.noise_sim, 1, 1, 1))
                        label_shape = tf.shape(label_noise)
                        label_noise = tf.reshape(label_noise, (label_shape[0]*label_shape[1], label_shape[2], label_shape[3], label_shape[4]))

                        action_label_tile = tf.reshape(tf.tile(tf.expand_dims(ACTION_LABEL, dim=1), (1, FLAGS.noise_sim, 1)), (FLAGS.batch_size * FLAGS.noise_sim, -1))

                        energy_noise = -FLAGS.temperature * model.forward(x_mod_neg, weights, label=label_noise, action_label=action_label_tile, reuse=True)
                        energy_noise = tf.reshape(energy_noise, (x_mod_neg_shape[0], FLAGS.noise_sim))
                        loss_energy_noise = model.forward(x_mod_neg, weights, label=label_noise, action_label=action_label_tile, reuse=True, stop_grad=True)

                        energy_wt = tf.nn.softmax(energy_noise, axis=1)
                        # energy_wt = tf.Print(energy_wt, [energy_wt])
                        energy_wt = tf.reshape(energy_wt, (x_mod_neg_shape[0], FLAGS.noise_sim, 1, 1, 1))
                        loss_energy_wt = tf.reshape(energy_wt, (-1, 1))
                        x_mod = tf.reduce_sum(energy_wt * x_mod_stack, axis=1)
                        x_grad = tf.zeros(1)

                        loss_energy_neg = loss_energy_noise * loss_energy_wt

                        loss_energys.append(loss_energy_neg)
                else:
                    # Use HMC to sample the distribution
                    if FLAGS.seq_update:
                        for n in range(FLAGS.pred_frame):
                            x_grad = tf.gradients(energy_noise, [x_mod])[0]

                            lr =  FLAGS.step_lr
                            x_mod_new = x_mod - lr * x_grad

                            if i != FLAGS.num_steps-1:
                                _mod_new = x_mod_new + tf.random_normal(tf.shape(x_mod), mean=0.0, stddev=1e-3)

                            idx = n
                            x_mod = tf.concat([x_mod[:, :idx], x_mod_new[:, idx:idx+1], x_mod[:, idx+1:]], axis=1)
                            energy_noise = model.forward(x_mod, weights, label=LABEL_SPLIT[j], action_label=ACTION_LABEL, reuse=True, stop_at_grad=True)

                        x_last = x_mod - (lr) * x_grad

                    else:
                        lr =  FLAGS.step_lr

                        x_grad = tf.gradients(FLAGS.temperature * energy_noise, [x_mod])[0]
                        energy_joint_noise = model.forward(x_joint_mod, weights, label=label_joint_mod, action_label=ACTION_LABEL, reuse=True)
                        x_joint_grad, label_joint_grad = tf.gradients(energy_joint_noise, [x_joint_mod, label_joint_mod])

                        if FLAGS.second_order:
                            EPS_CHANGE = 1e-5
                            x_mod_low = x_mod - EPS_CHANGE * x_grad
                            energy_noise_low = model.forward(x_mod_low, weights, label=LABEL_SPLIT[j], action_label=ACTION_LABEL, reuse=True)
                            x_grad_low = tf.gradients(FLAGS.temperature * energy_noise_low, [x_mod_low])[0]
                            x_grad_diff = (x_grad - x_grad_low) / EPS_CHANGE / x_grad
                            x_diff = lr * x_grad / tf.stop_gradient(tf.maximum(x_grad_diff, 1e-5))
                            x_mod = x_mod - x_diff

                        elif FLAGS.second_seq_opt:
                            batch_size = tf.shape(x_mod)[0]
                            # x_mod_compact = tf.reshape(x_mod, (batch_size, FLAGS.pred_frame * FLAGS.input_objects * FLAGS.latent_dim))
                            x_hess = tf.hessians(FLAGS.temperature * energy_noise, x_mod)[0]
                            compact_n = FLAGS.pred_frame*FLAGS.input_objects*FLAGS.latent_dim
                            # x_hess = tf.Print(x_hess, [tf.reduce_sum(tf.abs(x_hess))])
                            x_hess = tf.reshape(tf.reduce_sum(x_hess, axis=4), (batch_size, compact_n, compact_n))
                            x_grad_compact = tf.reshape(x_grad, (batch_size, compact_n, 1))
                            x_grad_compact_inv = tf.linalg.cholesky_solve(x_hess + 1e-3, x_grad_compact)
                            # x_grad_compact_inv = tf.nn.l2_normalize(x_grad_compact_inv, axis=1)
                            x_grad = tf.reshape(x_grad_compact_inv, (batch_size, FLAGS.pred_frame, FLAGS.input_objects, FLAGS.latent_dim))
                            # x_grad = tf.Print(x_grad, [x_grad])
                            x_mod = x_mod - lr * x_grad
                        else:
                            x_mod = x_mod - lr * x_grad

                        x_joint_mod = x_joint_mod - (lr) * x_joint_grad
                        label_joint_mod = label_joint_mod - (lr) * label_joint_grad

                        if FLAGS.proj_norm != 0.0:
                            if FLAGS.proj_norm_type == 'l2':
                                x_grad = tf.clip_by_norm(x_grad, FLAGS.proj_norm)
                            elif FLAGS.proj_norm_type == 'li':
                                x_grad = tf.clip_by_value(x_grad, -FLAGS.proj_norm, FLAGS.proj_norm)
                            else:
                                print("Other types of projection are not supported!!!")
                                assert False

                            x_grad = tf.stop_gradient(x_grad)

                        # energy_noise = tf.reshape(energy_noise, (-1, 1, 1, 1))
                        x_last = x_mod - (lr) * x_grad

                        x_ee = tf.stop_gradient(x_mod) - energy_noise * (lr) * x_grad
                        x_ees.append(x_ee)

                        x_mod = x_last

                        if i != FLAGS.num_steps-1:
                            x_mod = x_mod + tf.random_normal(tf.shape(x_mod), mean=0.0, stddev=0.001)

                        energy_noise = model.forward(x_mod, weights, label=LABEL_SPLIT[j], action_label=ACTION_LABEL, reuse=True, stop_at_grad=True)
                    loss_energys.append(model.forward(x_mod, weights, label=LABEL_SPLIT[j], action_label=ACTION_LABEL, reuse=True, stop_grad=True))

                x_grads.append(x_grad)
                x_mods.append(x_mod)

                energy_negs.append(model.forward(tf.stop_gradient(x_mod), weights, label=LABEL_SPLIT[j], action_label=ACTION_LABEL, reuse=True))
                print("Building loop {} ...".format(i))

            target_vars['x_mods'] = x_mods
            temp = FLAGS.temperature

            if FLAGS.gen_network:
                x_mod = gen_model.forward(LABEL_SPLIT[j], weights_other, action_label=ACTION_LABEL, reuse=False)

            energy_neg = model.forward(tf.stop_gradient(x_mod), weights, label=LABEL_SPLIT[j], action_label=ACTION_LABEL, reuse=True)
            loss_energy = tf.clip_by_value(temp * model.forward(x_mod, weights, reuse=True, label=LABEL, action_label=ACTION_LABEL, stop_grad=True), -1e5, 1e5)

            x_off = tf.reduce_mean(tf.square(x_mod - X_SPLIT[j]))

            print("Finished processing loop construction ...")

            if FLAGS.train:

                # energy_neg = tf.concat(energy_negs, axis=0)

                if FLAGS.gen_network:
                    # loss_ml = FLAGS.ml_coeff * tf.clip_by_value(tf.nn.softplus(temp * (energy_pos - energy_neg)), -1e5, 1e5)
                    # loss_total = tf.reduce_mean(loss_ml) + tf.reduce_mean(loss_energy) + tf.reduce_mean(tf.square(energy_pos)) + tf.reduce_mean(tf.square(energy_neg))

                    loss_ml = tf.zeros(1)
                    loss_total = tf.reduce_mean(tf.square(x_mod - X))
                elif not FLAGS.supervised:

                    # energy_neg = tf.concat(energy_negs[-1], axis=0)
                    energy_neg = energy_negs[-1]
                    if FLAGS.cd:
                        loss_ml = FLAGS.ml_coeff * FLAGS.temperature * (tf.reduce_mean(energy_pos) - tf.reduce_mean(energy_neg))
                        if FLAGS.zero_kl:
                            loss_energy = tf.zeros(1)
                    else:
                        energy_neg_reduced = (energy_neg - tf.reduce_min(energy_neg))
                        coeff = tf.stop_gradient(tf.exp(-temp*energy_neg_reduced))
                        norm_constant = tf.stop_gradient(tf.reduce_sum(coeff))
                        neg_loss = coeff * (-1*temp*energy_neg) / norm_constant
                        loss_ml = FLAGS.ml_coeff * (tf.reduce_mean(temp * energy_pos) + tf.reduce_sum(neg_loss))


                    loss_total = tf.reduce_mean(loss_ml) + tf.reduce_mean(loss_energy) + FLAGS.reg_scale * (tf.reduce_mean(tf.square(FLAGS.temperature * energy_pos)) + tf.reduce_mean(tf.square(FLAGS.temperature * energy_neg)))
                else:
                    x_mod = tf.stack(x_mods[-1:], axis=0)
                    X_expand = tf.expand_dims(X, axis=0)
                    loss_ml = tf.zeros(1)
                    # loss_total = tf.reduce_mean(tf.square(tf.stack(x_mods, axis=0) - X))
                    loss_total = tf.reduce_mean(tf.square(x_mod - X_expand))

                loss_ee = tf.zeros(1)
                loss_gp = tf.zeros(1)

                if FLAGS.ee:
                    x_ee = tf.stack(x_ees[:3], axis=1)
                    x_last = tf.expand_dims(x_last, axis=1)
                    loss_ee = FLAGS.ee_coeff * tf.reduce_mean(tf.square(tf.clip_by_value(x_ee - tf.stop_gradient(x_last), -1.0, 1.0)))
                    loss_total = loss_total + loss_ee

                if FLAGS.gp:
                    x_grads = tf.stack(x_grads, axis=0)
                    grads = tf.reshape(x_grads, [tf.shape(x_grads)[0], -1])
                    grads_norm = tf.norm(grads, axis=1)
                    loss_gp = FLAGS.gp_coeff * tf.reduce_mean(tf.square(grads_norm - 1))
                    loss_total = loss_total + loss_gp


                print("Started gradient computation...")
                gvs = optimizer.compute_gradients(loss_total)
                gvs = [(k, v) for (k, v) in gvs if k is not None]
                print("Applying gradients...")
                grads, vs = zip(*gvs)
                # capped_grads, _ = tf.clip_by_global_norm(grads, 1000)
                # gvs = list(zip(capped_grads, vs))

                def filter_grad(g, v):
                    return tf.clip_by_value(g, -1e5, 1e5)

                capped_gvs = [(filter_grad(grad, var), var) for grad, var in gvs]
                gvs = capped_gvs
                tower_grads.append(gvs)

                train_op = optimizer.apply_gradients(gvs)
                print("Finished applying gradients.")

                target_vars['loss_ml'] = loss_ml
                target_vars['loss_ee'] = loss_ee
                target_vars['loss_gp'] = loss_gp
                target_vars['total_loss'] = loss_total
                target_vars['gvs'] = gvs
                target_vars['loss_energy'] = loss_energy
                target_vars['weights'] = weights


            target_vars['X'] = X
            target_vars['LABEL'] = LABEL
            target_vars['X_NOISE'] = X_NOISE
            target_vars['energy_pos'] = energy_pos
            target_vars['energy_start'] = energy_start
            target_vars['x_grad'] = x_grad
            target_vars['x_grad_first'] = x_grads[0]
            target_vars['x_mod'] = x_mod
            target_vars['x_off'] = x_off
            target_vars['temp'] = temp
            target_vars['energy_neg'] = energy_negs[-1]
            target_vars['test_x_mod'] = x_mod
            target_vars['lr'] = LR
            target_vars['x_joint_mod'] = x_joint_mod
            target_vars['label_joint_mod'] = label_joint_mod
            target_vars['ACTION_LABEL'] = ACTION_LABEL

    if FLAGS.train:
        grads = average_gradients(tower_grads)
        train_op = optimizer.apply_gradients(grads)
        target_vars['train_op'] = train_op

    sess = tf.InteractiveSession()
    # saver = loader = tf.train.Saver(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES), max_to_keep=10)
    saver = loader = tf.train.Saver(max_to_keep=10,  keep_checkpoint_every_n_hours=2)

    total_parameters = 0
    for variable in tf.trainable_variables():
        # shape is an array of tf.Dimension
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

    if FLAGS.datasource == 'fetch':
        dataset = np.load('fetch.npz')['obs'][:, :, None, :]
        actions = np.load('fetch.npz')['action']
    elif FLAGS.datasource == 'hand':
        dataset = np.load('hand.npz')['obs'][:, :, None, :]
        dataset = dataset[:, :, :, :]
        actions = np.load('hand.npz')['action']

    dataset_flat = dataset.reshape((-1, dataset.shape[-1]))
    mean, std = dataset_flat.mean(axis=0), dataset_flat.std(axis=0)
    std = std + 1e-5

    dataset = (dataset - mean) / std
    split_idx = int(dataset.shape[0] * 0.9)

    dataset_train = dataset[:split_idx]
    actions_train = actions[:split_idx]
    dataset_test = dataset[split_idx:]
    actions_test = actions[split_idx:]

    print(dataset_test[0, 0, 0])
    if FLAGS.datasource == 'hand':
        # Scale the hand coordinates
        # dataset[:, :, :, 24:31] *= 10
        # dataset *= 10
        pass


    if FLAGS.single:
        dataset = np.tile(dataset[0:1], (100, 1, 1, 1))[:, :20]

    if FLAGS.train:
        train(target_vars, saver, sess, logger, dataset_train, actions_train, resume_itr)

    test(target_vars, saver, sess, logdir, dataset_test, actions_test, mean, std, dataset_train)

if __name__ == "__main__":
    main()
