""" Utility functions. """
import os
import random
import warnings

import numpy as np
import tensorflow as tf
from scipy import linalg
from tensorflow.contrib.layers.python import layers as tf_layers
from tensorflow.core.util import event_pb2
from tensorflow.python.platform import flags

FLAGS = flags.FLAGS
flags.DEFINE_integer('spec_iter', 1, 'Number of iterations to normalize spectrum of matrix')
flags.DEFINE_float('spec_norm_val', 1.0, 'Desired norm of matrices')
flags.DEFINE_bool('downsample', False, 'Wheter to do average pool downsampling')
flags.DEFINE_bool('spec_eval', False, 'Set to true to prevent spectral updates')

def safemean(xs):
    return np.nan if len(xs) == 0 else np.mean(xs)


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


def compute_lr(itr, lr):
    frac = min((itr + 100) / 300, 1)
    return frac * lr


def log_step_num_exp(d):
    import csv
    with open('get_avg_step_num_log.csv', mode='a+') as csv_file:
        fieldnames = list(d.keys())
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writerow(d)


def parse_valid_obs(obs, actions, dones):
    """Given obs, actions, and dones array, return a set of valid transitions"""

    # Literally just do a for loop through the obs to generate the actions needed
    t_obs = []
    t_actions = []

    for i in range(obs.shape[0]):
        for t in range(1, obs.shape[1]):
            if not dones[i, t - 1]:
                pair_obs = obs[i, t - 1:t + 1]
                action = actions[i, t - 1]

                t_obs.append(pair_obs)
                t_actions.append(action)

    t_actions = np.array(t_actions)
    t_obs = np.array(t_obs)

    return t_actions, t_obs


def get_median(v):
    v = tf.reshape(v, [-1])
    m = tf.shape(v)[0] // 2
    return tf.nn.top_k(v, m)[m - 1]


def set_seed(seed):
    import torch
    import numpy
    import random

    torch.manual_seed(seed)
    numpy.random.seed(seed)
    random.seed(seed)
    tf.set_random_seed(seed)


class ReplayBuffer(object):
    def __init__(self, size):
        """Create Replay buffer.
        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self._storage = []
        self._maxsize = size
        self._next_idx = 0

    def __len__(self):
        return len(self._storage)

    def add(self, ims):
        batch_size = ims.shape[0]
        if self._next_idx >= len(self._storage):
            self._storage.extend(list(ims))
        else:
            if batch_size + self._next_idx < self._maxsize:
                self._storage[self._next_idx:self._next_idx + batch_size] = list(ims)
            else:
                split_idx = self._maxsize - self._next_idx
                self._storage[self._next_idx:] = list(ims)[:split_idx]
                self._storage[:batch_size - split_idx] = list(ims)[split_idx:]
        self._next_idx = (self._next_idx + ims.shape[0]) % self._maxsize

    def _encode_sample(self, idxes):
        ims = []
        for i in idxes:
            ims.append(self._storage[i])
        return np.array(ims)

    def sample(self, batch_size):
        """Sample a batch of experiences.
        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        """
        idxes = [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
        return self._encode_sample(idxes)


## Setting weights for equalized learning rates
def get_weight(name, shape, gain=np.sqrt(2), use_wscale=False, fan_in=None, spec_norm=False, zero=False, fc=False):
    if fan_in is None: fan_in = np.prod(shape[:-1])
    std = gain / np.sqrt(fan_in)  # He init
    if use_wscale:
        wscale = tf.constant(np.float32(std), name=name + 'wscale')
        var = tf.get_variable(name + 'weight', shape=shape, initializer=tf.initializers.random_normal()) * wscale
    elif spec_norm:
        if zero:
            var = tf.get_variable(shape=shape, name=name + 'weight',
                                  initializer=tf.initializers.random_normal(stddev=1e-10))
            var = spectral_normed_weight(var, name, lower_bound=True, fc=fc)
        else:
            var = tf.get_variable(name + 'weight', shape=shape, initializer=tf.initializers.random_normal())
            var = spectral_normed_weight(var, name, fc=fc)
    else:
        if zero:
            var = tf.get_variable(name + 'weight', shape=shape, initializer=tf.initializers.zero())
        else:
            var = tf.get_variable(name + 'weight', shape=shape,
                                  initializer=tf.contrib.layers.xavier_initializer(dtype=tf.float32))

    return var


def pixel_norm(x, epsilon=1e-8):
    with tf.variable_scope('PixelNorm'):
        return x * tf.rsqrt(tf.reduce_mean(tf.square(x), axis=[1, 2], keepdims=True) + epsilon)


##helper
def get_images(paths, labels, nb_samples=None, shuffle=True):
    if nb_samples is not None:
        sampler = lambda x: random.sample(x, nb_samples)
    else:
        sampler = lambda x: x
    images = [(i, os.path.join(path, image)) \
              for i, path in zip(labels, paths) \
              for image in sampler(os.listdir(path))]
    if shuffle:
        random.shuffle(images)
    return images


def optimistic_restore(session, save_file):
    reader = tf.train.NewCheckpointReader(save_file)
    saved_shapes = reader.get_variable_to_shape_map()
    var_names = sorted([(var.name, var.name.split(':')[0]) for var in tf.global_variables()
                        if var.name.split(':')[0] in saved_shapes])
    restore_vars = []
    with tf.variable_scope('', reuse=True):
        for var_name, saved_var_name in var_names:
            try:
                curr_var = tf.get_variable(saved_var_name)
            except Exception as e:
                print(e)
                continue
            var_shape = curr_var.get_shape().as_list()
            if var_shape == saved_shapes[saved_var_name]:
                restore_vars.append(curr_var)
    print(restore_vars)
    saver = tf.train.Saver(restore_vars)
    saver.restore(session, save_file)


def remap_restore(session, save_file, i):
    reader = tf.train.NewCheckpointReader(save_file)
    saved_shapes = reader.get_variable_to_shape_map()
    var_names = sorted([(var.name, var.name.split(':')[0]) for var in tf.global_variables()
                        if var.name.split(':')[0] in saved_shapes])
    restore_vars = []
    with tf.variable_scope('', reuse=True):
        for var_name, saved_var_name in var_names:
            try:
                curr_var = tf.get_variable(saved_var_name)
            except Exception as e:
                print(e)
                continue
            var_shape = curr_var.get_shape().as_list()
            if var_shape == saved_shapes[saved_var_name]:
                restore_vars.append(curr_var)
    print(restore_vars)
    saver = tf.train.Saver(restore_vars)
    saver.restore(session, save_file)


## Network weight initializers
def init_conv_weight(weights, scope, k, c_in, c_out, spec_norm=True, zero=False, scale=1.0, classes=1):
    if spec_norm:
        spec_norm = FLAGS.spec_norm

    conv_weights = {}
    with tf.variable_scope(scope):
        if zero:
            conv_weights['c'] = get_weight('c', [k, k, c_in, c_out], spec_norm=spec_norm, zero=True)
            # conv_weights['c'] = tf.Variable(tf.zeros([k, k, c_in, c_out]), name='c')
        else:
            conv_weights['c'] = get_weight('c', [k, k, c_in, c_out], spec_norm=spec_norm)

        # conv_weights['c'] = conv_weights['c'] * scale

        conv_weights['b'] = tf.get_variable(shape=[c_out], name='b', initializer=tf.initializers.zeros())

        if FLAGS.cclass:
            conv_weights['g'] = tf.get_variable(shape=[classes, c_out], name='g', initializer=tf.initializers.ones())
            conv_weights['gb'] = tf.get_variable(shape=[classes, c_in], name='gb', initializer=tf.initializers.zeros())
        else:
            conv_weights['g'] = tf.get_variable(shape=[c_out], name='g', initializer=tf.initializers.ones())
            conv_weights['gb'] = tf.get_variable(shape=[c_in], name='gb', initializer=tf.initializers.zeros())

    weights[scope] = conv_weights


def init_attention_weight(weights, scope, c_in, k, trainable_gamma=True, spec_norm=True):
    if spec_norm:
        spec_norm = FLAGS.spec_norm

    atten_weights = {}
    with tf.variable_scope(scope):
        atten_weights['q'] = get_weight('atten_q', [1, 1, c_in, k], spec_norm=spec_norm)
        atten_weights['q_b'] = tf.get_variable(shape=[k], name='atten_q_b1', initializer=tf.initializers.zeros())
        atten_weights['k'] = get_weight('atten_k', [1, 1, c_in, k], spec_norm=spec_norm)
        atten_weights['k_b'] = tf.get_variable(shape=[k], name='atten_k_b1', initializer=tf.initializers.zeros())
        atten_weights['v'] = get_weight('atten_v', [1, 1, c_in, c_in], spec_norm=spec_norm)
        atten_weights['v_b'] = tf.get_variable(shape=[c_in], name='atten_v_b1', initializer=tf.initializers.zeros())
        atten_weights['gamma'] = tf.get_variable(shape=[1], name='gamma', initializer=tf.initializers.zeros())

    weights[scope] = atten_weights


def init_fc_weight(weights, scope, c_in, c_out, spec_norm=True):
    fc_weights = {}

    if spec_norm:
        spec_norm = FLAGS.spec_norm

    with tf.variable_scope(scope):
        fc_weights['w'] = get_weight('w', [c_in, c_out], spec_norm=spec_norm, fc=True)
        fc_weights['b'] = tf.get_variable(shape=[c_out], name='b', initializer=tf.initializers.zeros())

    weights[scope] = fc_weights


def init_res_weight(weights, scope, k, c_in, c_out, hidden_dim=None, spec_norm=True, res_scale=1.0, classes=1):
    if not hidden_dim:
        hidden_dim = c_in

    if spec_norm:
        spec_norm = FLAGS.spec_norm

    init_conv_weight(weights, scope + '_res_c1', k, c_in, c_out, spec_norm=spec_norm, scale=res_scale, classes=classes)
    init_conv_weight(weights, scope + '_res_c2', k, c_out, c_out, spec_norm=spec_norm, zero=True, scale=res_scale,
                     classes=classes)

    if c_in != c_out:
        init_conv_weight(weights, scope + '_res_adaptive', k, c_in, c_out, spec_norm=spec_norm, scale=res_scale,
                         classes=classes)


## Network forward helpers
def smart_conv_block(inp, weights, reuse, scope, use_stride=True, **kwargs):
    weights = weights[scope]
    return conv_block(inp, weights['c'], weights['b'], reuse, scope, scale=weights['g'], bias=weights['gb'],
                      use_stride=use_stride, **kwargs)


def smart_res_block(inp, weights, reuse, scope, downsample=True, adaptive=True, stop_batch=False, upsample=False,
                    label=None, **kwargs):
    gn1 = weights[scope + '_res_c1']
    gn2 = weights[scope + '_res_c2']
    # inp = group_norm(inp, gn1['g'], gn1['gb'], stop_batch=stop_batch)
    # inp_act = tf.nn.leaky_relu(inp)
    c1 = smart_conv_block(inp, weights, reuse, scope + '_res_c1', use_stride=False, activation=None, extra_bias=True,
                          label=label, **kwargs)
    # c1 = group_norm(c1, gn2['g'], gn2['gb'], stop_batch=stop_batch)
    c1 = tf.nn.leaky_relu(c1)
    c2 = smart_conv_block(c1, weights, reuse, scope + '_res_c2', use_stride=False, activation=None, use_scale=True,
                          extra_bias=True, label=label, **kwargs)

    if adaptive:
        c_bypass = smart_conv_block(inp, weights, reuse, scope + '_res_adaptive', use_stride=False, activation=None,
                                    **kwargs)
    else:
        c_bypass = inp

    # c_bypass = tf.Print(c_bypass, [c_bypass], message=scope)

    res = c2 + c_bypass

    if upsample:
        res_shape = tf.shape(res)
        res_shape_list = res.get_shape()
        res = tf.image.resize_nearest_neighbor(res, [2 * res_shape_list[1], 2 * res_shape_list[2]])
        # res = tf.image.resize_images(res, [2*res_shape_list[1], 2*res_shape_list[2]])
    elif downsample:
        res = tf.nn.avg_pool(res, (1, 2, 2, 1), (1, 2, 2, 1), 'VALID')

    res = tf.nn.leaky_relu(res)

    return res


def smart_res_block_optim(inp, weights, reuse, scope, **kwargs):
    c1 = smart_conv_block(inp, weights, reuse, scope + '_res_c1', use_stride=False, activation=None, **kwargs)
    c1 = tf.nn.leaky_relu(c1)
    c2 = smart_conv_block(c1, weights, reuse, scope + '_res_c2', use_stride=False, activation=None, **kwargs)

    inp = tf.nn.avg_pool(inp, (1, 2, 2, 1), (1, 2, 2, 1), 'VALID')
    c_bypass = smart_conv_block(inp, weights, reuse, scope + '_res_adaptive', use_stride=False, activation=None,
                                **kwargs)
    c2 = tf.nn.avg_pool(c2, (1, 2, 2, 1), (1, 2, 2, 1), 'VALID')

    res = c2 + c_bypass

    return c2


def smart_atten_block(inp, weights, reuse, scope, **kwargs):
    w = weights[scope]
    return attention(inp, w['q'], w['q_b'], w['k'], w['k_b'], w['v'], w['v_b'], w['gamma'], reuse, scope, **kwargs)


def smart_fc_block(inp, weights, reuse, scope, use_bias=True):
    weights = weights[scope]
    output = tf.matmul(inp, weights['w'])

    if use_bias:
        output = output + weights['b']

    return output


## Network helpers
def conv_block(inp, cweight, bweight, reuse, scope, use_stride=True, activation=tf.nn.leaky_relu, pn=False,
               bn=False, gn=False, ln=False, scale=None, bias=None, use_bias=False, downsample=False, stop_batch=False,
               use_scale=False, extra_bias=False, average=False, label=None):
    """ Perform, conv, batch norm, nonlinearity, and max pool """
    stride, no_stride = [1, 2, 2, 1], [1, 1, 1, 1]
    _, h, w, _ = inp.get_shape()

    if FLAGS.downsample:
        stride = no_stride

    if not FLAGS.use_bias and not use_bias:
        bweight = 0

    if extra_bias:
        if label is not None:
            bias_batch = tf.matmul(label, bias)
            batch = tf.shape(bias_batch)[0]
            dim = tf.shape(bias_batch)[1]
            bias = tf.reshape(bias_batch, (batch, 1, 1, dim))

        inp = inp + bias

    if not use_stride:
        conv_output = tf.nn.conv2d(inp, cweight, no_stride, 'SAME')
    else:
        conv_output = tf.nn.conv2d(inp, cweight, stride, 'SAME')

    if use_scale:
        if label is not None:
            scale_batch = tf.matmul(label, scale)
            batch = tf.shape(scale_batch)[0]
            dim = tf.shape(scale_batch)[1]
            scale = tf.reshape(scale_batch, (batch, 1, 1, dim))

        conv_output = conv_output * scale

    if use_bias:
        conv_output = conv_output + bweight

    if activation is not None:
        conv_output = activation(conv_output)

    if bn:
        conv_output = batch_norm(conv_output, scale, bias)
    if pn:
        conv_output = pixel_norm(conv_output)
    if gn:
        conv_output = group_norm(conv_output, scale, bias, stop_batch=stop_batch)
    if ln:
        conv_output = layer_norm(conv_output, scale, bias)

    if FLAGS.downsample and use_stride:
        conv_output = tf.layers.average_pooling2d(conv_output, (2, 2), 2)

    return conv_output


def conv_block_1d(inp, cweight, bweight, reuse, scope, activation=tf.nn.leaky_relu):
    """ Perform, conv, batch norm, nonlinearity, and max pool """
    stride = 1

    conv_output = tf.nn.conv1d(inp, cweight, stride, 'SAME') + bweight

    if activation is not None:
        conv_output = activation(conv_output)

    return conv_output


def conv_block_3d(inp, cweight, bweight, reuse, scope, use_stride=True, activation=tf.nn.leaky_relu, pn=False,
                  bn=False, gn=False, ln=False, scale=None, bias=None, use_bias=False):
    """ Perform, conv, batch norm, nonlinearity, and max pool """
    stride, no_stride = [1, 1, 2, 2, 1], [1, 1, 1, 1, 1]
    _, d, h, w, _ = inp.get_shape()

    if not FLAGS.use_bias and not use_bias:
        bweight = 0

    if not use_stride:
        conv_output = tf.nn.conv3d(inp, cweight, no_stride, 'SAME') + bweight
    else:
        conv_output = tf.nn.conv3d(inp, cweight, stride, 'SAME') + bweight

    if activation is not None:
        conv_output = activation(conv_output, alpha=0.1)

    if bn:
        conv_output = batch_norm(conv_output, scale, bias)
    if pn:
        conv_output = pixel_norm(conv_output)
    if gn:
        conv_output = group_norm(conv_output, scale, bias)
    if ln:
        conv_output = layer_norm(conv_output, scale, bias)

    if FLAGS.downsample and use_stride:
        conv_output = tf.layers.average_pooling2d(conv_output, (2, 2), 2)

    return conv_output


def group_norm(inp, scale, bias, g=32, eps=1e-6, stop_batch=False):
    """Applies group normalization assuming nhwc format"""
    n, h, w, c = inp.shape
    inp = tf.reshape(inp, (tf.shape(inp)[0], h, w, c // g, g))

    mean, var = tf.nn.moments(inp, [1, 2, 4], keep_dims=True)
    gain = tf.rsqrt(var + eps)

    # if stop_batch:
    #     gain = tf.stop_gradient(gain)

    output = gain * (inp - mean)
    output = tf.reshape(output, (tf.shape(inp)[0], h, w, c))

    if scale is not None:
        output = output * scale

    if bias is not None:
        output = output + bias

    return output


def layer_norm(inp, scale, bias, eps=1e-6):
    """Applies group normalization assuming nhwc format"""
    n, h, w, c = inp.shape

    mean, var = tf.nn.moments(inp, [1, 2, 3], keep_dims=True)
    gain = tf.rsqrt(var + eps)
    output = gain * (inp - mean)

    if scale is not None:
        output = output * scale

    if bias is not None:
        output = output + bias

    return output


def conv_cond_concat(x, y):
    """Concatenate conditioning vector on feature map axis."""
    x_shapes = tf.shape(x)
    y_shapes = tf.shape(y)

    return tf.concat([x, y * tf.ones([x_shapes[0], x_shapes[1], x_shapes[2], y_shapes[3]]) / 10.], 3)


def attention(inp, q, q_b, k, k_b, v, v_b, gamma, reuse, scope, stop_at_grad=False, seperate=False, scale=False):
    conv_q = conv_block(inp, q, q_b, reuse=reuse, scope=scope, use_stride=False, activation=None, use_bias=True,
                        pn=False, bn=False, gn=False)
    conv_k = conv_block(inp, k, k_b, reuse=reuse, scope=scope, use_stride=False, activation=None, use_bias=True,
                        pn=False, bn=False, gn=False)

    if stop_at_grad and (not FLAGS.ignore_stop_at_grad):
        conv_k = tf.stop_gradient(conv_k)
        conv_q = tf.stop_gradient(conv_q)

    conv_v = conv_block(inp, v, v_b, reuse=reuse, scope=scope, use_stride=False, pn=False, bn=False, gn=False)

    c_num = float(conv_q.get_shape().as_list()[-1])
    s = tf.matmul(hw_flatten(conv_q), hw_flatten(conv_k), transpose_b=True)

    if scale:
        s = s / (c_num) ** 0.5
    beta = tf.nn.softmax(s, axis=-1)
    o = tf.matmul(beta, hw_flatten(conv_v))
    o = tf.reshape(o, shape=tf.shape(inp))
    # inp = tf.Print(inp, [conv_q, conv_k, beta])
    inp = inp + gamma * o

    if not seperate:
        return inp
    else:
        return gamma * o


def attention_2d(inp, q, q_b, k, k_b, v, v_b, reuse, scope, stop_at_grad=False, seperate=False, scale=False):
    inp_shape = tf.shape(inp)
    inp_compact = tf.reshape(inp, (inp_shape[0] * FLAGS.input_objects * inp_shape[1], inp.shape[3]))
    f_q = tf.matmul(inp_compact, q) + q_b
    f_k = tf.matmul(inp_compact, k) + k_b
    f_v = tf.nn.leaky_relu(tf.matmul(inp_compact, v) + v_b)

    f_q = tf.reshape(f_q, (inp_shape[0], inp_shape[1], inp_shape[2], tf.shape(f_q)[-1]))
    f_k = tf.reshape(f_k, (inp_shape[0], inp_shape[1], inp_shape[2], tf.shape(f_k)[-1]))
    f_v = tf.reshape(f_v, (inp_shape[0], inp_shape[1], inp_shape[2], inp_shape[3]))

    # if stop_at_grad:
    #     f_q = tf.stop_gradient(f_q)
    #     f_k = tf.stop_gradient(f_k)

    s = tf.matmul(f_k, f_q, transpose_b=True)
    c_num = (32 ** 0.5)

    if scale:
        s = s / c_num

    beta = tf.nn.softmax(s, axis=-1)

    o = tf.reshape(tf.matmul(beta, f_v), inp_shape)

    return o


def hw_flatten(x):
    shape = tf.shape(x)
    return tf.reshape(x, [tf.shape(x)[0], -1, shape[-1]])


def batch_norm(inp, scale, bias, eps=0.01):
    mean, var = tf.nn.moments(inp, [0])
    output = tf.nn.batch_normalization(inp, mean, var, bias, scale, eps)
    return output


def normalize(inp, activation, reuse, scope):
    if FLAGS.norm == 'batch_norm':
        return tf_layers.batch_norm(inp, activation_fn=activation, reuse=reuse, scope=scope)
    elif FLAGS.norm == 'layer_norm':
        return tf_layers.layer_norm(inp, activation_fn=activation, reuse=reuse, scope=scope)
    elif FLAGS.norm == 'None':
        if activation is not None:
            return activation(inp)
        else:
            return inp


## Loss functions
def mse(pred, label):
    pred = tf.reshape(pred, [-1])
    label = tf.reshape(label, [-1])
    return tf.reduce_mean(tf.square(pred - label))


def xent(pred, label):
    # Note - with tf version <=0.12, this loss has incorrect 2nd derivatives
    return tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=label) / FLAGS.update_batch_size


NO_OPS = 'NO_OPS'


def _l2normalize(v, eps=1e-12):
    return v / (tf.reduce_sum(v ** 2) ** 0.5 + eps)


def spectral_normed_weight(w, name, lower_bound=False, iteration=1, fc=False):
    if fc:
        iteration = 2

    w_shape = w.shape.as_list()
    w = tf.reshape(w, [-1, w_shape[-1]])

    iteration = FLAGS.spec_iter
    sigma_new = FLAGS.spec_norm_val

    u = tf.get_variable(name + "_u", [1, w_shape[-1]], initializer=tf.random_normal_initializer(), trainable=False)

    u_hat = u
    v_hat = None
    for i in range(iteration):
        """
        power iteration
        Usually iteration = 1 will be enough
        """
        v_ = tf.matmul(u_hat, tf.transpose(w))
        v_hat = tf.nn.l2_normalize(v_)

        u_ = tf.matmul(v_hat, w)
        u_hat = tf.nn.l2_normalize(u_)

    u_hat = tf.stop_gradient(u_hat)
    v_hat = tf.stop_gradient(v_hat)

    sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))
    # sigma = tf.Print(sigma, [sigma])

    if FLAGS.spec_eval:
        dep = []
    else:
        dep = [u.assign(u_hat)]

    with tf.control_dependencies(dep):
        if lower_bound:
            sigma = sigma + 1e-6
            w_norm = w / sigma * tf.minimum(sigma, 1) * sigma_new
        else:
            w_norm = w / sigma * sigma_new

        w_norm = tf.reshape(w_norm, w_shape)

    return w_norm


# def spectral_normed_weight(W, name, u=None, num_iters=1, update_collection=None, with_sigma=False, sigma_new=1, lower_bound=False):
#     # Usually num_iters = 1 will be enough
#     num_iters = FLAGS.spec_iter
#     sigma_new = FLAGS.spec_norm_val
#     W_shape = W.shape.as_list()
#     W_reshaped = tf.reshape(W, [-1, W_shape[-1]])
# 
#     if u is None:
#         u = tf.get_variable(name+"_u", [1, W_shape[-1]], initializer=tf.truncated_normal_initializer(), trainable=False)
# 
#     def power_iteration(i, u_i, v_i):
#         v_ip1 = _l2normalize(tf.matmul(u_i, tf.transpose(W_reshaped)))
#         u_ip1 = _l2normalize(tf.matmul(v_ip1, W_reshaped))
#         return i + 1, u_ip1, v_ip1
# 
#     _, u_final, v_final = tf.while_loop(
#       cond=lambda i, _1, _2: i < num_iters,
#       body=power_iteration,
#       loop_vars=(tf.constant(0, dtype=tf.int32),
#                  u, tf.zeros(dtype=tf.float32, shape=[1, W_reshaped.shape.as_list()[0]]))
#     )
# 
#     if update_collection is None:
#         warnings.warn('Setting update_collection to None will make u being updated every W execution. This maybe undesirable'
#                       '. Please consider using a update collection instead.')
#         # sigma = tf.matmul(tf.matmul(v_final, W_reshaped), tf.transpose(u_final))[0, 0] + 1e-6
#         sigma = tf.reduce_sum(tf.matmul(u_final, tf.transpose(W_reshaped)) * v_final) + 1e-6
#         if lower_bound:
#             W_bar = W_reshaped / sigma * tf.minimum(tf.abs(sigma), 1) * sigma_new
#         else:
#             W_bar = W_reshaped / sigma * sigma_new
# 
#         with tf.control_dependencies([u.assign(u_final)]):
#             W_bar = tf.reshape(W_bar, W_shape)
#     else:
#         sigma = tf.matmul(tf.matmul(v_final, W_reshaped), tf.transpose(u_final))[0, 0]
#         # sigma = tf.reduce_sum(tf.matmul(u_final, tf.transpose(W_reshaped)) * v_final)
# 
#         if lower_bound:
#             W_bar = W_reshaped / sigmaA * tf.minimum(tf.abs(sigmaA), 1) * sigma_new
#         else:
#             W_bar = W_reshaped / sigmaA * sigma_new
#         W_bar = tf.reshape(W_bar, W_shape)
#         # Put NO_OPS to not update any collection. This is useful for the second call of discriminator if the update_op
#         # has already been collected on the first call.
# 
#         if update_collection != NO_OPS:
#             tf.add_to_collection(update_collection, u.assign(u_final))
# 
#     if with_sigma:
#         return W_bar, sigma
#     else:
#         return W_bar


def average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.
    Note that this function provides a synchronization point across all towers.
    Args:
      tower_grads: List of lists of (gradient, variable) tuples. The outer list
        is over individual gradients. The inner list is over the gradient
        calculation for each tower.
    Returns:
       List of pairs of (gradient, variable) where the gradient has been averaged
       across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, v in grad_and_vars:
            if g is not None:
                # Add 0 dimension to the gradients to represent the tower.
                expanded_g = tf.expand_dims(g, 0)

                # Append on a 'tower' dimension which we will average over below.
                grads.append(expanded_g)
            else:
                print(g, v)

        # Average over the 'tower' dimension.
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


def norm_phys_data(arr):
    # Expects array with last dimension of shape 6, first two as velocity, next two are position, and final
    # are angular monetum

    arr = arr.copy()
    arr[:, :, :, :2] = arr[:, :, :, :2] / 400 * 6
    arr[:, :, :, 2:4] = (arr[:, :, :, 2:4] / 84 - 0.5) * 4
    arr[:, :, :, 4:6] = arr[:, :, :, 4:6] / 140 * 8

    return arr


def unnorm_phys_data(arr):
    # Expects array with last dimension of shape 6, first two as velocity, next two are position, and final
    # are angular monetum
    arr = arr.copy()

    if arr.shape[2] == 6:
        arr[:, :, :2] = arr[:, :, :2] * 400 / 6
        arr[:, :, 2:4] = (arr[:, :, 2:4] * 0.25 + 0.5) * 84
        arr[:, :, 4:6] = arr[:, :, 4:6] * 140 / 8
    else:
        arr[:, :, :] = (arr[:, :, :] * 0.25 + 0.5) * 84

    return arr


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.

    Params:
    -- mu1 : Numpy array containing the activations of the pool_3 layer of the
             inception net ( like returned by the function 'get_predictions')
             for generated samples.
    -- mu2   : The sample mean over activations of the pool_3 layer, precalcualted
               on an representive data set.
    -- sigma1: The covariance matrix over activations of the pool_3 layer for
               generated samples.
    -- sigma2: The covariance matrix over activations of the pool_3 layer,
               precalcualted on an representive data set.

    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, "Training and test mean vectors have different lengths"
    assert sigma1.shape == sigma2.shape, "Training and test covariances have different dimensions"

    diff = mu1 - mu2

    # product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = "fid calculation produces singular product; adding %s to diagonal of cov estimates" % eps
        warnings.warn(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.

    Params:
    -- mu1 : Numpy array containing the activations of the pool_3 layer of the
             inception net ( like returned by the function 'get_predictions')
             for generated samples.
    -- mu2   : The sample mean over activations of the pool_3 layer, precalcualted
               on an representive data set.
    -- sigma1: The covariance matrix over activations of the pool_3 layer for
               generated samples.
    -- sigma2: The covariance matrix over activations of the pool_3 layer,
               precalcualted on an representive data set.

    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, "Training and test mean vectors have different lengths"
    assert sigma1.shape == sigma2.shape, "Training and test covariances have different dimensions"

    diff = mu1 - mu2

    # product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = "fid calculation produces singular product; adding %s to diagonal of cov estimates" % eps
        warnings.warn(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


def is_maze_valid(dat):
    # Generate an indicator function for whether a point is valid in a hand designed
    # maze given dat (n x 2) array of data_points

    segs = 4
    oob_mask = np.any(oob(dat), axis=1)
    # dat = dat[~data_mask]

    dat_idx = ((dat[:, 0] + 1) * segs).astype(np.int32)

    data_mask = ((dat_idx % 2) == 0) | (((dat_idx % 4) == 1) & (dat[:, 1] > 0.7)) | (
            ((dat_idx % 4) == 3) & (dat[:, 1] < -0.7))

    comb_mask = (~oob_mask) & data_mask

    return comb_mask
