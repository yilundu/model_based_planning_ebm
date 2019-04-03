import tensorflow as tf
from tensorflow.python.platform import flags
import numpy as np
from utils import conv_block, get_weight, attention, conv_cond_concat, conv_block_3d, attention_2d, conv_block_1d
from utils import smart_fc_block, init_fc_weight
# from rl_common.tf_util import huber_loss

FLAGS = flags.FLAGS


class TrajNetLatent(object):
    """Construct the convolutional network specified in MAML"""

    def __init__(self, dim_input=6, num_filters=64, dim_output=1, action_dim=20):

        self.dim_hidden = num_filters
        self.dim_output = dim_output
        self.dim_input = dim_input
        self.img_size = FLAGS.im_size
        self.action_dim = action_dim

    def construct_weights(self, scope='', action_size = 0):
        weights = {}
        dtype = tf.float32
        conv_initializer =  tf.contrib.layers.xavier_initializer_conv2d(dtype=dtype)
        fc_initializer =  tf.contrib.layers.xavier_initializer(dtype=dtype)

        with tf.variable_scope(scope):
            weights['w_upscale'] = get_weight('w_up', [self.dim_input, self.dim_hidden], spec_norm=FLAGS.spec_norm)
            weights['b_upscale'] = tf.Variable(tf.zeros([self.dim_hidden]), name='b_upscale')

            weights['atten_q'] = get_weight('atten_q', [self.dim_hidden, self.dim_hidden], spec_norm=FLAGS.spec_norm)
            weights['atten_q_b'] = tf.Variable(tf.zeros([self.dim_hidden]), name='atten_q_b')

            weights['atten_k'] = get_weight('atten_k', [self.dim_hidden, self.dim_hidden], spec_norm=FLAGS.spec_norm)
            weights['atten_k_b'] = tf.Variable(tf.zeros([self.dim_hidden]), name='atten_k_b')

            weights['atten_v'] = get_weight('attten_v', [self.dim_hidden, self.dim_hidden], spec_norm=FLAGS.spec_norm)
            weights['atten_v_b'] = tf.Variable(tf.zeros([self.dim_hidden]), name='atten_v_b')

            weights['conv1'] = get_weight('conv1', [4, FLAGS.input_objects*self.dim_hidden, self.dim_hidden], spec_norm=FLAGS.spec_norm)
            weights['b1'] = tf.Variable(tf.zeros([self.dim_hidden]), name='b1')

            weights['conv1a'] = get_weight('conv1a', [4, self.dim_hidden, self.dim_hidden], spec_norm=FLAGS.spec_norm)
            weights['b1a'] = tf.Variable(tf.zeros([self.dim_hidden]), name='b1a')

            weights['w1'] = get_weight('w1', [self.dim_hidden, self.dim_hidden], spec_norm=FLAGS.spec_norm)
            weights['b2'] = tf.Variable(tf.zeros([self.dim_hidden]), name='b2')

            weights['w1'] = get_weight('w1b', [self.dim_hidden, self.dim_hidden], spec_norm=FLAGS.spec_norm)
            weights['b2'] = tf.Variable(tf.zeros([self.dim_hidden]), name='bb')

            weights['w2'] = get_weight('w2', [self.dim_hidden, 1], spec_norm=FLAGS.spec_norm)
            weights['b3'] = tf.Variable(tf.zeros([1]), name='b3')

            if action_size != 0:
                weights['action_w'] = get_weight('action_w', [self.action_dim, self.dim_hidden], spec_norm=FLAGS.spec_norm)
                weights['action_b'] = get_weight('action_b', [self.action_dim, self.dim_hidden], spec_norm=FLAGS.spec_norm)

                weights['action_w_2'] = get_weight('action_w_2', [self.action_dim, self.dim_hidden], spec_norm=FLAGS.spec_norm)
                weights['action_b_2'] = tf.Variable(tf.zeros([self.dim_hidden]), name='action_b_2')

        return weights

    def forward(self, inp, weights, label, action_label=None, reuse=False, scope='', stop_grad=False, stop_at_grad=False, noise=True, return_logit=False):
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

        if action_label is not None:
            weight_action = tf.tile(tf.matmul(action_label, weights['action_w']), (joint_shape[1] * FLAGS.input_objects, 1))
            weight_bias = tf.tile(tf.matmul(action_label, weights['action_b']), (joint_shape[1] * FLAGS.input_objects, 1))
            joint_compact = joint_compact * weight_action + weight_bias

        joint = tf.reshape(joint_compact, (joint_shape[0], joint_shape[1], joint_shape[2], self.dim_hidden))

        joint = attention_2d(joint, weights['atten_q'], weights['atten_q_b'], weights['atten_k'], weights['atten_k_b'], weights['atten_v'], weights['atten_v_b'], reuse, scope+'atten', stop_at_grad=stop_at_grad)

        hidden1 = tf.reshape(joint, (batch_size, traj_len, FLAGS.input_objects * self.dim_hidden))
        hidden1 = conv_block_1d(hidden1, weights['conv1'], weights['b1'], reuse, scope+'0', activation=act)
        hidden1 = conv_block_1d(hidden1, weights['conv1a'], weights['b1a'], reuse, scope+'1', activation=act)
        hidden1 = tf.reduce_mean(hidden1, axis=1)



        hidden2 = tf.nn.leaky_relu(tf.matmul(hidden1, weights['w1']) + weights['b2'])
        energy = tf.matmul(hidden1, weights['w2']) + weights['b3']
        return energy


class TrajNetLatentFC(object):
    """Construct the convolutional network specified in MAML"""

    def __init__(self, dim_input=6, num_filters=128, dim_output=1, action_dim=20):

        self.dim_hidden = num_filters
        self.dim_output = dim_output
        self.dim_input = dim_input
        self.img_size = FLAGS.im_size

    def construct_weights(self, scope='', action_size = 0):
        weights = {}
        dtype = tf.float32
        conv_initializer =  tf.contrib.layers.xavier_initializer_conv2d(dtype=dtype)
        fc_initializer =  tf.contrib.layers.xavier_initializer(dtype=dtype)

        if FLAGS.no_cond:
            action_size = 0

        with tf.variable_scope(scope):
            weights['w1'] = get_weight('w1', [FLAGS.input_objects*self.dim_input*(FLAGS.label_frame+FLAGS.pred_frame) + action_size, self.dim_hidden], spec_norm=FLAGS.spec_norm)
            weights['b1'] = tf.Variable(tf.zeros([self.dim_hidden]), name='b1')
            weights['w2'] = get_weight('w2', [self.dim_hidden, self.dim_hidden], spec_norm=FLAGS.spec_norm)
            weights['b2'] = tf.Variable(tf.zeros([self.dim_hidden]), name='b2')
            weights['w3'] = get_weight('w3', [self.dim_hidden, self.dim_hidden], spec_norm=FLAGS.spec_norm)
            weights['b3'] = tf.Variable(tf.zeros([self.dim_hidden]), name='b3')
            weights['w3a'] = get_weight('w3a', [self.dim_hidden, self.dim_hidden], spec_norm=FLAGS.spec_norm)
            weights['b3a'] = tf.Variable(tf.zeros([self.dim_hidden]), name='b3a')
            weights['w4'] = get_weight('w4', [self.dim_hidden, self.dim_hidden], spec_norm=FLAGS.spec_norm)
            weights['b4'] = tf.Variable(tf.zeros([self.dim_hidden]), name='b4')
            weights['w5'] = get_weight('w5', [self.dim_hidden, self.dim_hidden], spec_norm=FLAGS.spec_norm)
            weights['b5'] = tf.Variable(tf.zeros([self.dim_hidden]), name='b5')
            weights['w6'] = get_weight('w6', [self.dim_hidden, 1], spec_norm=FLAGS.spec_norm)

            # if action_size != 0:
            #     weights['action_w'] = get_weight('action_w', [action_size, self.dim_hidden], spec_norm=FLAGS.spec_norm)
            #     weights['action_b'] = tf.Variable(tf.zeros([self.dim_hidden]), name='action_b')

        return weights

    def forward(self, inp, weights, label, reuse=False, scope='', stop_grad=False, stop_at_grad=False, noise=True, action_label=False):
        weights = weights.copy()
        batch_size = tf.shape(inp)[0]

        joint = tf.concat([label, inp], axis=1)

        joint = tf.reshape(joint, (-1, FLAGS.input_objects*self.dim_input*(FLAGS.label_frame+FLAGS.pred_frame)))

        if action_label is not None and (not FLAGS.no_cond):
            joint = tf.concat([joint, action_label], axis=1)

        h1 = tf.nn.leaky_relu(tf.matmul(joint, weights['w1']) + weights['b1'])
        h2 = tf.nn.leaky_relu(tf.matmul(h1, weights['w2']) + weights['b2'])

        # if action_label is not None:
        #     weight_gain = tf.matmul(action_label, weights['action_w']) + weights['action_b']
        #     h2 = h2 * weight_gain

        h3 = tf.nn.leaky_relu(tf.matmul(h2, weights['w3']) + weights['b3'])
        # h4 = tf.nn.leaky_relu(tf.matmul(h3, weights['w3a']) + weights['b3a'])
        # h5 = tf.nn.leaky_relu(tf.matmul(h4, weights['w4']) + weights['b4'])
        # h6 = tf.nn.leaky_relu(tf.matmul(h5, weights['w5']) + weights['b5'])
        energy = tf.matmul(h3, weights['w6'])

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

    def __init__(self, dim_input=6, num_filters=256, dim_output=1, action_dim=20):

        self.dim_hidden = num_filters
        self.dim_output = dim_output
        self.dim_input = dim_input
        self.img_size = FLAGS.im_size

    def construct_weights(self, scope='', action_size=0):
        weights = {}
        dtype = tf.float32
        conv_initializer =  tf.contrib.layers.xavier_initializer_conv2d(dtype=dtype)
        fc_initializer =  tf.contrib.layers.xavier_initializer(dtype=dtype)

        if FLAGS.no_cond:
            action_size = 0

        with tf.variable_scope(scope):
            init_fc_weight(weights, 'fc1', FLAGS.input_objects*self.dim_input*(FLAGS.label_frame) + action_size, 128)
            init_fc_weight(weights, 'fc2', 128, 128)
            init_fc_weight(weights, 'fc4', 128, 128)
            init_fc_weight(weights, 'fc3', 128, FLAGS.input_objects*self.dim_input*(FLAGS.pred_frame))

            if action_size != 0:
                weights['action_w_fc'] = get_weight('action_w_fc', [action_size, 128], spec_norm=FLAGS.spec_norm)
                weights['action_b_fc'] = tf.Variable(tf.zeros([128]), name='action_b_fc')

        return weights

    def forward(self, label, weights, reuse=False, scope='', stop_grad=False, stop_at_grad=False, noise=True, action_label=None):
        weights = weights.copy()
        batch_size = tf.shape(label)[0]
        label_flat = tf.reshape(label, (batch_size, FLAGS.input_objects*self.dim_input*FLAGS.label_frame))

        if action_label is not None and (not FLAGS.no_cond):
            label_flat = tf.concat([label_flat, action_label], axis=1)

        h1 = tf.nn.leaky_relu(smart_fc_block(label_flat, weights, reuse, 'fc1'))
        h2 = tf.nn.leaky_relu(smart_fc_block(h1, weights, reuse, 'fc2'))
        h3 = tf.nn.leaky_relu(smart_fc_block(h2, weights, reuse, 'fc4'))

        # if action_label is not None:
        #     weight_gain = tf.matmul(action_label, weights['action_w_fc']) + weights['action_b_fc']
        #     h2 = h2 * weight_gain

        output_label = smart_fc_block(h3, weights, reuse, 'fc3')
        output_label = tf.reshape(output_label, (batch_size, FLAGS.pred_frame, FLAGS.input_objects, self.dim_input))

        return output_label


class TrajNetLatentGen(object):
    """Construct the convolutional network specified in MAML"""

    def __init__(self, dim_input=6, num_filters=32, dim_output=1):

        self.dim_hidden = num_filters
        self.dim_output = dim_output
        self.dim_input = dim_input
        self.img_size = FLAGS.im_size

    def construct_weights(self, scope='', action_size=0):
        weights = {}
        dtype = tf.float32
        conv_initializer =  tf.contrib.layers.xavier_initializer_conv2d(dtype=dtype)
        fc_initializer =  tf.contrib.layers.xavier_initializer(dtype=dtype)

        self.action_dim = action_size

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

            weights['w1'] = get_weight('w1', [4*self.dim_hidden, 4*self.dim_hidden])
            weights['b2'] = tf.Variable(tf.zeros([4*self.dim_hidden]), name='b2')

            n = FLAGS.input_objects * self.dim_input * FLAGS.pred_frame
            weights['w2'] = get_weight('w2', [4*self.dim_hidden, n], spec_norm=FLAGS.spec_norm)
            weights['b3'] = tf.Variable(tf.zeros([n]), name='b3')

            if action_size != 0:
                weights['action_w'] = get_weight('action_w', [self.action_dim, 4*self.dim_hidden], spec_norm=FLAGS.spec_norm)
                weights['action_b'] = tf.Variable(tf.zeros([4*self.dim_hidden]), name='action_b')

        return weights

    def forward(self, inp, weights, action_label=None, reuse=False, scope='', stop_grad=False, stop_at_grad=False, noise=True):
        act = tf.nn.leaky_relu

        weights = weights.copy()
        batch_size = tf.shape(inp)[0]
        traj_len = tf.shape(inp)[1]

        joint = inp

        if stop_grad:
            for k, v in weights.items():
                weights[k] = tf.stop_gradient(v)

        joint_shape = tf.shape(joint)
        joint_compact = tf.reshape(joint, (joint_shape[0] * FLAGS.input_objects * joint_shape[1], joint_shape[3]))
        joint_compact = act(tf.matmul(joint_compact, weights['w_upscale']) + weights['b_upscale'])
        joint = tf.reshape(joint_compact, (joint_shape[0], joint_shape[1], joint_shape[2], self.dim_hidden))

        hidden1 = joint
        hidden1 = attention_2d(joint, weights['atten_q'], weights['atten_q_b'], weights['atten_k'], weights['atten_k_b'], weights['atten_v'], weights['atten_v_b'], reuse, scope+'atten', stop_at_grad=stop_at_grad)

        hidden1 = tf.reshape(hidden1, (batch_size, traj_len, FLAGS.input_objects * self.dim_hidden))
        hidden1 = conv_block_1d(hidden1, weights['conv1'], weights['b1'], reuse, scope+'0', activation=act)
        hidden1 = tf.reduce_mean(hidden1, axis=1)

        if action_label is not None:
            weight_action = tf.matmul(action_label, weights['action_w']) + weights['action_b']
            hidden1 = hidden1 * weight_action

        hidden2 = act(tf.matmul(hidden1, weights['w1']) + weights['b2'])
        pred_seq = tf.matmul(hidden2, weights['w2']) + weights['b3']
        pred_seq = tf.reshape(pred_seq, (batch_size, 1, 1, self.dim_input))

        return pred_seq
