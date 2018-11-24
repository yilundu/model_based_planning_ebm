import tensorflow as tf
from tensorflow.python.platform import flags
import numpy as np
from utils import conv_block, get_weight, attention, conv_cond_concat, conv_block_3d, attention_2d, conv_block_1d
from utils import smart_fc_block, init_fc_weight
# from rl_common.tf_util import huber_loss

FLAGS = flags.FLAGS


class TrajNetLatent(object):
    """Construct the convolutional network specified in MAML"""

    def __init__(self, dim_input=6, num_filters=32, dim_output=1):

        self.dim_hidden = num_filters
        self.dim_output = dim_output
        self.dim_input = dim_input
        self.img_size = FLAGS.im_size

    def construct_weights(self, scope=''):
        weights = {}
        dtype = tf.float32
        conv_initializer =  tf.contrib.layers.xavier_initializer_conv2d(dtype=dtype)
        fc_initializer =  tf.contrib.layers.xavier_initializer(dtype=dtype)

        with tf.variable_scope(scope):
            weights['w_upscale'] = get_weight('w_up', [self.dim_input, self.dim_hidden], spec_norm=FLAGS.spec_norm)
            weights['b_upscale'] = tf.Variable(tf.zeros([self.dim_hidden]), name='b_upscale')

            weights['atten_q'] = get_weight('atten_q', [self.dim_hidden, self.dim_hidden / 2], spec_norm=FLAGS.spec_norm)
            weights['atten_q_b'] = tf.Variable(tf.zeros([self.dim_hidden / 2]), name='atten_q_b')

            weights['atten_k'] = get_weight('atten_k', [self.dim_hidden, self.dim_hidden / 2], spec_norm=FLAGS.spec_norm)
            weights['atten_k_b'] = tf.Variable(tf.zeros([self.dim_hidden / 2]), name='atten_k_b')

            weights['atten_v'] = get_weight('attten_v', [self.dim_hidden, self.dim_hidden], spec_norm=FLAGS.spec_norm)
            weights['atten_v_b'] = tf.Variable(tf.zeros([self.dim_hidden]), name='atten_v_b')

            weights['conv1'] = get_weight('conv1', [5, FLAGS.input_objects*self.dim_hidden, 4*self.dim_hidden], spec_norm=FLAGS.spec_norm)
            weights['b1'] = tf.Variable(tf.zeros([4*self.dim_hidden]), name='b1')

            weights['conv2'] = get_weight('conv2', [5, 4*self.dim_hidden, 4*self.dim_hidden], spec_norm=FLAGS.spec_norm)
            weights['b2'] = tf.Variable(tf.zeros([4*self.dim_hidden]), name='b2')

            weights['w1'] = get_weight('w1', [4*self.dim_hidden, 1], spec_norm=FLAGS.spec_norm)
            weights['b3'] = tf.Variable(tf.zeros([1]), name='b3')

            # weights['w2'] = get_weight('w2', [128, 1])

        return weights

    def forward(self, inp, weights, label, reuse=False, scope='', stop_grad=False, stop_at_grad=False, noise=True, return_logit=False):
        # if FLAGS.second_seq_opt:
        #     act = tf.nn.elu
        # else:

        # if FLAGS.grad_free:
        #     act = tf.nn.elu
        # else:
        act = tf.nn.leaky_relu

        weights = weights.copy()
        batch_size = tf.shape(inp)[0]

        joint = tf.concat([label, inp], axis=1)

        # print("Joint shape, ", joint.get_shape())

        traj_len = tf.shape(joint)[1]

        if stop_grad:
            for k, v in weights.items():
                weights[k] = tf.stop_gradient(v)

        joint_shape = tf.shape(joint)
        joint_compact = tf.reshape(joint, (joint_shape[0] * FLAGS.input_objects * joint_shape[1], joint_shape[3]))
        joint_compact = act(tf.matmul(joint_compact, weights['w_upscale']) + weights['b_upscale'])
        joint = tf.reshape(joint_compact, (joint_shape[0], joint_shape[1], joint_shape[2], self.dim_hidden))

        hidden1 = attention_2d(joint, weights['atten_q'], weights['atten_q_b'], weights['atten_k'], weights['atten_k_b'], weights['atten_v'], weights['atten_v_b'], reuse, scope+'atten', stop_at_grad=stop_at_grad)

        hidden1 = tf.reshape(hidden1, (batch_size, traj_len, FLAGS.input_objects * self.dim_hidden))
        hidden2 = conv_block_1d(hidden1, weights['conv1'], weights['b1'], reuse=reuse, scope='c1', activation=act)
        latent = hidden2 = conv_block_1d(hidden2, weights['conv2'], weights['b2'], reuse=reuse, scope='c2', activation=act)

        hidden2_shape = tf.shape(hidden2)
        hidden2 = tf.reshape(hidden2, (batch_size * hidden2_shape[1], 4*self.dim_hidden))
        hidden3 = act(tf.matmul(hidden2, weights['w1']) + weights['b3'])
        # hidden4 = tf.matmul(hidden3, weights['w2'])
        hidden3 = tf.reshape(hidden3, (batch_size, hidden2_shape[1]))

        energy = tf.reduce_sum(hidden3, [1])

        if return_logit:
            return tf.reshape(latent, (batch_size, hidden2_shape[1] * 4 * self.dim_hidden))
        else:
            return energy


class TrajNetLatentFC(object):
    """Construct the convolutional network specified in MAML"""

    def __init__(self, dim_input=6, num_filters=256, dim_output=1):

        self.dim_hidden = num_filters
        self.dim_output = dim_output
        self.dim_input = dim_input
        self.img_size = FLAGS.im_size

    def construct_weights(self, scope=''):
        weights = {}
        dtype = tf.float32
        conv_initializer =  tf.contrib.layers.xavier_initializer_conv2d(dtype=dtype)
        fc_initializer =  tf.contrib.layers.xavier_initializer(dtype=dtype)

        with tf.variable_scope(scope):
            weights['w1'] = get_weight('w1', [FLAGS.input_objects*self.dim_input*(FLAGS.label_frame+FLAGS.pred_frame), self.dim_hidden], spec_norm=FLAGS.spec_norm)
            weights['b1'] = tf.Variable(tf.zeros([self.dim_hidden]), name='b1')
            weights['w2'] = get_weight('w2', [self.dim_hidden, self.dim_hidden], spec_norm=FLAGS.spec_norm)
            weights['b2'] = tf.Variable(tf.zeros([self.dim_hidden]), name='b2')
            weights['w3'] = get_weight('w3', [self.dim_hidden, self.dim_hidden], spec_norm=FLAGS.spec_norm)
            weights['b3'] = tf.Variable(tf.zeros([self.dim_hidden]), name='b3')
            weights['w4'] = get_weight('w4', [self.dim_hidden, 1], spec_norm=FLAGS.spec_norm)

        return weights

    def forward(self, inp, weights, label, reuse=False, scope='', stop_grad=False, stop_at_grad=False, noise=True):
        weights = weights.copy()
        batch_size = tf.shape(inp)[0]
        joint = tf.concat([label, inp], axis=1)

        joint = tf.reshape(joint, (-1, FLAGS.input_objects*self.dim_input*(FLAGS.label_frame+FLAGS.pred_frame)))

        h1 = tf.nn.leaky_relu(tf.matmul(joint, weights['w1']) + weights['b1'])
        h2 = tf.nn.leaky_relu(tf.matmul(h1, weights['w2']) + weights['b2'])
        h3 = tf.nn.leaky_relu(tf.matmul(h2, weights['w3']) + weights['b3'])
        energy = tf.matmul(h3, weights['w4'])

        return energy


class TrajNetLatentDir(object):
    """Construct the convolutional network specified in MAML"""

    def __init__(self, dim_input=6, num_filters=32, dim_output=1):

        self.dim_hidden = num_filters
        self.dim_output = dim_output
        self.dim_input = dim_input
        self.img_size = FLAGS.im_size

    def construct_weights(self, scope=''):
        weights = {}
        dtype = tf.float32
        conv_initializer =  tf.contrib.layers.xavier_initializer_conv2d(dtype=dtype)
        fc_initializer =  tf.contrib.layers.xavier_initializer(dtype=dtype)

        with tf.variable_scope(scope):
            weights['w_upscale'] = get_weight('w_up', [self.dim_input, self.dim_hidden], spec_norm=FLAGS.spec_norm)
            weights['b_upscale'] = tf.Variable(tf.zeros([self.dim_hidden]), name='b_upscale')

            weights['atten_q'] = get_weight('atten_q', [self.dim_hidden, 32], spec_norm=FLAGS.spec_norm)
            weights['atten_q_b'] = tf.Variable(tf.zeros([32]), name='atten_q_b')

            weights['atten_k'] = get_weight('atten_k', [self.dim_hidden, 32], spec_norm=FLAGS.spec_norm)
            weights['atten_k_b'] = tf.Variable(tf.zeros([32]), name='atten_k_b')

            weights['atten_v'] = get_weight('attten_v', [self.dim_hidden, self.dim_hidden], spec_norm=FLAGS.spec_norm)
            weights['atten_v_b'] = tf.Variable(tf.zeros([self.dim_hidden]), name='atten_v_b')

            # weights['conv1'] = get_weight('conv1', [5, FLAGS.input_objects*self.dim_hidden, 4*self.dim_hidden], spec_norm=FLAGS.spec_norm)
            # weights['b1'] = tf.Variable(tf.zeros([4*self.dim_hidden]), name='b1')

            # weights['conv2'] = get_weight('conv2', [5, 4*self.dim_hidden, 4*self.dim_hidden], spec_norm=FLAGS.spec_norm)
            # weights['b2'] = tf.Variable(tf.zeros([4*self.dim_hidden]), name='b2')

            weights['w1'] = get_weight('w1', [FLAGS.input_objects*self.dim_hidden*(FLAGS.pred_frame+FLAGS.label_frame), 512], spec_norm=FLAGS.spec_norm)
            weights['b1'] = tf.Variable(tf.zeros([512]), name='b1')
            weights['w2'] = get_weight('w2', [512, 512], spec_norm=FLAGS.spec_norm)
            weights['b2'] = tf.Variable(tf.zeros([512]), name='b2')
            weights['w3'] = get_weight('w2', [512, 1])
            weights['b3'] = tf.Variable(tf.zeros([1]), name='b3')

        return weights

    def forward(self, inp, weights, label, reuse=False, scope='', stop_grad=False, stop_at_grad=False, noise=True):
        weights = weights.copy()
        batch_size = tf.shape(inp)[0]

        joint = tf.concat([label, inp], axis=1)

        traj_len = tf.shape(joint)[1]

        if stop_grad:
            for k, v in weights.items():
                weights[k] = tf.stop_gradient(v)

        joint_shape = tf.shape(joint)
        joint_compact = tf.reshape(joint, (joint_shape[0] * FLAGS.input_objects * joint_shape[1], joint_shape[3]))
        joint_compact = tf.nn.leaky_relu(tf.matmul(joint_compact, weights['w_upscale']) + weights['b_upscale'])
        joint = tf.reshape(joint_compact, (joint_shape[0], joint_shape[1], joint_shape[2], self.dim_hidden))

        hidden1 = attention_2d(joint, weights['atten_q'], weights['atten_q_b'], weights['atten_k'], weights['atten_k_b'], weights['atten_v'], weights['atten_v_b'], reuse, scope+'atten', stop_at_grad=stop_at_grad)

        hidden1 = tf.reshape(joint, (batch_size, traj_len, FLAGS.input_objects * self.dim_hidden))
        # hidden2 = conv_block_1d(hidden1, weights['conv1'], weights['b1'], reuse=reuse, scope='c1')
        # hidden2 = conv_block_1d(hidden2, weights['conv2'], weights['b2'], reuse=reuse, scope='c2')

        # hidden2_shape = tf.shape(hidden2)
        # hidden2 = tf.reshape(hidden2, (batch_size * hidden2_shape[1], 4*self.dim_hidden))
        hidden2 = tf.reshape(hidden1, (batch_size, -1))

        hidden3 = tf.nn.leaky_relu(tf.matmul(hidden2, weights['w1']) + weights['b1'])
        hidden4 = tf.nn.leaky_relu(tf.matmul(hidden3, weights['w2']) + weights['b2'])
        energy = tf.nn.leaky_relu(tf.matmul(hidden4, weights['w3']) + weights['b3'])
        # hidden4 = tf.matmul(hidden3, weights['w2'])
        # hidden4 = tf.reshape(hidden4, (batch_size, hidden2_shape[1]))

        # energy = tf.reduce_mean(hidden4, [1]) + weights['b4']

        return energy


class TrajNetLatentGenFC(object):
    """Construct the convolutional network specified in MAML"""

    def __init__(self, dim_input=6, num_filters=256, dim_output=1):

        self.dim_hidden = num_filters
        self.dim_output = dim_output
        self.dim_input = dim_input
        self.img_size = FLAGS.im_size

    def construct_weights(self, scope=''):
        weights = {}
        dtype = tf.float32
        conv_initializer =  tf.contrib.layers.xavier_initializer_conv2d(dtype=dtype)
        fc_initializer =  tf.contrib.layers.xavier_initializer(dtype=dtype)

        with tf.variable_scope(scope):
            init_fc_weight(weights, 'fc1', FLAGS.input_objects*self.dim_input*(FLAGS.label_frame), 128, spec_norm=True)
            init_fc_weight(weights, 'fc2', 128, 128, spec_norm=True)
            init_fc_weight(weights, 'fc3', 128, FLAGS.input_objects*self.dim_input*(FLAGS.pred_frame), spec_norm=True)

        return weights

    def forward(self, label, weights, reuse=False, scope='', stop_grad=False, stop_at_grad=False, noise=True):
        weights = weights.copy()
        batch_size = tf.shape(label)[0]
        label_flat = tf.reshape(label, (batch_size, FLAGS.input_objects*self.dim_input*FLAGS.label_frame))

        h1 = tf.nn.leaky_relu(smart_fc_block(label_flat, weights, reuse, 'fc1'))
        h2 = tf.nn.leaky_relu(smart_fc_block(h1, weights, reuse, 'fc2'))
        output_label = smart_fc_block(h2, weights, reuse, 'fc3')

        output_label = tf.reshape(output_label, (batch_size, FLAGS.pred_frame, FLAGS.input_objects, self.dim_input))

        return output_label


class TrajNetLatentGen(object):
    """Construct the convolutional network specified in MAML"""

    def __init__(self, dim_input=6, num_filters=32, dim_output=1):

        self.dim_hidden = num_filters
        self.dim_output = dim_output
        self.dim_input = dim_input
        self.img_size = FLAGS.im_size

    def construct_weights(self, scope=''):
        weights = {}
        dtype = tf.float32
        conv_initializer =  tf.contrib.layers.xavier_initializer_conv2d(dtype=dtype)
        fc_initializer =  tf.contrib.layers.xavier_initializer(dtype=dtype)

        with tf.variable_scope('gen_' + scope):
            weights['w_upscale'] = get_weight('w_up', [self.dim_input, self.dim_hidden], spec_norm=FLAGS.spec_norm)
            weights['b_upscale'] = tf.Variable(tf.zeros([self.dim_hidden]), name='b_upscale')

            weights['atten_q'] = get_weight('atten_q', [self.dim_hidden, self.dim_hidden / 2], spec_norm=FLAGS.spec_norm)
            weights['atten_q_b'] = tf.Variable(tf.zeros([self.dim_hidden / 2]), name='atten_q_b')

            weights['atten_k'] = get_weight('atten_k', [self.dim_hidden, self.dim_hidden / 2], spec_norm=FLAGS.spec_norm)
            weights['atten_k_b'] = tf.Variable(tf.zeros([self.dim_hidden / 2]), name='atten_k_b')

            weights['atten_v'] = get_weight('attten_v', [self.dim_hidden, self.dim_hidden], spec_norm=FLAGS.spec_norm)
            weights['atten_v_b'] = tf.Variable(tf.zeros([self.dim_hidden]), name='atten_v_b')

            weights['conv1'] = get_weight('conv1', [5, FLAGS.input_objects*self.dim_hidden, 4*self.dim_hidden], spec_norm=FLAGS.spec_norm)
            weights['b1'] = tf.Variable(tf.zeros([4*self.dim_hidden]), name='b1')

            weights['conv2'] = get_weight('conv2', [5, 4*self.dim_hidden, 4*self.dim_hidden], spec_norm=FLAGS.spec_norm)
            weights['b2'] = tf.Variable(tf.zeros([4*self.dim_hidden]), name='b2')

            weights['w1'] = get_weight('w1', [4*self.dim_hidden, 128], spec_norm=FLAGS.spec_norm)
            weights['b3'] = tf.Variable(tf.zeros([128]), name='b3')

            weights['w2'] = get_weight('w2', [128, FLAGS.input_objects*self.dim_input*(FLAGS.pred_frame)])
            weights['b4'] = tf.Variable(tf.zeros([FLAGS.input_objects*self.dim_input*(FLAGS.pred_frame)]), name='b4')

        return weights

    def forward(self, label, weights, reuse=False, scope='', stop_grad=False, stop_at_grad=False, noise=True):
        weights = weights.copy()
        batch_size = tf.shape(label)[0]

        # if stop_grad:
        #     for k, v in weights.items():
        #         weights[k] = tf.stop_gradient(v)

        joint_shape = tf.shape(label)
        traj_len = joint_shape[1]
        joint_compact = tf.reshape(label, (joint_shape[0] * FLAGS.input_objects * joint_shape[1], joint_shape[3]))
        joint_compact = tf.nn.elu(tf.matmul(joint_compact, weights['w_upscale']) + weights['b_upscale'])
        joint = tf.reshape(joint_compact, (joint_shape[0], joint_shape[1], joint_shape[2], self.dim_hidden))

        hidden1 = attention_2d(joint, weights['atten_q'], weights['atten_q_b'], weights['atten_k'], weights['atten_k_b'], weights['atten_v'], weights['atten_v_b'], reuse, scope+'atten', stop_at_grad=stop_at_grad, scale=True)

        hidden1 = tf.reshape(hidden1, (batch_size, traj_len, FLAGS.input_objects * self.dim_hidden))
        hidden2 = conv_block_1d(hidden1, weights['conv1'], weights['b1'], reuse=reuse, scope='c1', activation=tf.nn.elu)
        hidden2 = conv_block_1d(hidden2, weights['conv2'], weights['b2'], reuse=reuse, scope='c2', activation=tf.nn.elu)

        hidden2_shape = tf.shape(hidden2)
        hidden2 = tf.reshape(hidden2, (batch_size * hidden2_shape[1], 4*self.dim_hidden))
        hidden3 = tf.nn.elu(tf.matmul(hidden2, weights['w1']) + weights['b3'])

        hidden4 = tf.reshape(hidden3, (batch_size, hidden2_shape[1], 128))
        mean_rep = tf.reduce_mean(hidden4, [1])

        hidden4 = tf.matmul(mean_rep, weights['w2']) + weights['b4']
        output_label = tf.reshape(hidden4, (batch_size, FLAGS.pred_frame, FLAGS.input_objects, self.dim_input))

        return output_label
