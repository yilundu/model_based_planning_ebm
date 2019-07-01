import tensorflow as tf
from tensorflow.python.platform import flags

from utils import get_weight, attention_2d, conv_block_1d

FLAGS = flags.FLAGS


class TrajNetLatent(object):
    """Construct the convolutional network specified in MAML"""

    def __init__(self, dim_input=6, num_filters=64, dim_output=1, action_dim=20):

        self.dim_hidden = num_filters
        self.dim_output = dim_output
        self.dim_input = dim_input
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

    def forward(self, inp, weights, action_label=None, reuse=False, scope='', stop_grad=False, stop_at_grad=False, noise=True, return_logit=False):
        act = tf.nn.leaky_relu
        weights = weights.copy()
        batch_size = tf.shape(inp)[0]
        traj_len = tf.shape(inp)[1]

        joint = inp

        if stop_grad:
            for k, v in weights.items():
                weights[k] = tf.stop_gradient(v)

        joint_shape = tf.shape(joint)
        joint_compact = inp
        joint_compact = act(tf.matmul(joint_compact, weights['w_upscale']) + weights['b_upscale'])

        if action_label is not None and FLAGS.cond:
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

    def construct_weights(self, scope='', action_size=2):
        weights = {}
        dtype = tf.float32
        conv_initializer =  tf.contrib.layers.xavier_initializer_conv2d(dtype=dtype)
        fc_initializer =  tf.contrib.layers.xavier_initializer(dtype=dtype)

        if not FLAGS.cond:
            action_size = 0

        with tf.variable_scope(scope):
            weights['w1'] = get_weight('w1', [FLAGS.input_objects*self.dim_input*(FLAGS.total_frame) + action_size,
                                              self.dim_hidden], spec_norm=FLAGS.spec_norm)
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
            weights['w6'] = get_weight('w6', [self.dim_hidden, 1], spec_norm=False, pos=True)

        return weights

    def forward(self, inp, weights, reuse=False, scope='', stop_grad=False, stop_at_grad=False, noise=True, action_label=False, opt_low=False):
        weights = weights.copy()
        batch_size = tf.shape(inp)[0]
        print(inp.get_shape())

        if FLAGS.datasource == "continual_reacher":
            # Set the goal position to be for stability
            inp = tf.concat([inp[:, :, :, :-3], tf.zeros(tf.shape(inp[:, :, :, -3:]))], axis=3)

        def swish(inp):
            return inp * tf.nn.sigmoid(inp)

        joint = inp
        joint = tf.reshape(joint, (-1, FLAGS.input_objects*self.dim_input*(FLAGS.total_frame)))

        if action_label is not None and (FLAGS.cond):
            joint = tf.concat([joint, action_label], axis=1)

        h1 = tf.nn.leaky_relu(tf.matmul(joint, weights['w1']) + weights['b1'])
        h2 = tf.nn.leaky_relu(tf.matmul(h1, weights['w2']) + weights['b2'])
        h3 = tf.nn.leaky_relu(tf.matmul(h2, weights['w3']) + weights['b3'])

        energy = tf.matmul(h3, weights['w6'])

        if FLAGS.opt_low:
            _, var = tf.nn.moments(h3, [1])
            std = tf.sqrt(tf.expand_dims(var, 1))
            # std = tf.Print(std, [std], "std")
            energy = energy - std
            # energy = tf.Print(energy, [energy, std], message="relative scale of energy/std")

        return energy


class TrajInverseDynamics(object):
    """Construct the convolutional network specified in MAML"""

    def __init__(self, dim_input=2, num_filters=32, dim_output=2, action_dim=20):

        self.dim_hidden = num_filters
        self.dim_output = dim_output
        self.dim_input = dim_input

    def construct_weights(self, scope='', weights={}, reuse=False):
        dtype = tf.float32
        conv_initializer = tf.contrib.layers.xavier_initializer_conv2d(dtype=dtype)
        fc_initializer = tf.contrib.layers.xavier_initializer(dtype=dtype)

        with tf.variable_scope(scope, reuse=reuse):
            weights['inv_w1'] = get_weight('w1',
                                       [FLAGS.input_objects * self.dim_input * (FLAGS.total_frame),
                                        self.dim_hidden], spec_norm=False)
            weights['inv_b1'] = tf.Variable(tf.zeros([self.dim_hidden]), name='b1')
            weights['inv_w2'] = get_weight('w2', [self.dim_hidden, self.dim_hidden], spec_norm=False)
            weights['inv_b2'] = tf.Variable(tf.zeros([self.dim_hidden]), name='b2')
            weights['inv_w3'] = get_weight('w3', [self.dim_hidden, self.dim_hidden], spec_norm=False)
            weights['inv_b3'] = tf.Variable(tf.zeros([self.dim_hidden]), name='b3')
            weights['inv_w6'] = get_weight('w6', [self.dim_hidden, self.dim_output], spec_norm=False)

        return weights

    def forward(self, inp, weights, reuse=False, scope='', stop_grad=False, stop_at_grad=False, noise=True,
                action_label=None):
        weights = weights.copy()
        batch_size = tf.shape(inp)[0]

        def swish(inp):
            return inp * tf.nn.sigmoid(inp)

        joint = inp
        joint = tf.reshape(joint, (-1, FLAGS.input_objects * self.dim_input * (FLAGS.total_frame)))

        if action_label is not None and FLAGS.cond:
            joint = tf.concat([joint, action_label], axis=1)

        h1 = tf.nn.relu(tf.matmul(joint, weights['inv_w1']) + weights['inv_b1'])
        h2 = tf.nn.relu(tf.matmul(h1, weights['inv_w2']) + weights['inv_b2'])
        h3 = (tf.matmul(h2, weights['inv_w3']) + weights['inv_b3'])
        energy = tf.matmul(h3, weights['inv_w6'])

        return energy


class TrajFFDynamics(object):
    """Construct FF network"""

    def __init__(self, dim_input=4, num_filters=128, dim_output=1, action_dim=20):

        self.dim_hidden = num_filters
        self.dim_output = dim_output
        self.dim_input = dim_input

    def construct_weights(self, scope='', weights={}, action_size=2):
        dtype = tf.float32
        conv_initializer = tf.contrib.layers.xavier_initializer_conv2d(dtype=dtype)
        fc_initializer = tf.contrib.layers.xavier_initializer(dtype=dtype)
        print(action_size)

        with tf.variable_scope(scope):
            weights['ff_w1'] = get_weight('w1',
                                       [FLAGS.input_objects * self.dim_input + action_size,
                                        self.dim_hidden], spec_norm=False)
            weights['ff_b1'] = tf.Variable(tf.zeros([self.dim_hidden]), name='b1')
            weights['ff_w2'] = get_weight('w2', [self.dim_hidden, self.dim_hidden], spec_norm=False)
            weights['ff_b2'] = tf.Variable(tf.zeros([self.dim_hidden]), name='b2')
            weights['ff_w3'] = get_weight('w3', [self.dim_hidden, self.dim_hidden], spec_norm=False)
            weights['ff_b3'] = tf.Variable(tf.zeros([self.dim_hidden]), name='b3')
            weights['ff_w6'] = get_weight('w6', [self.dim_hidden, self.dim_output], spec_norm=False)

        return weights

    def forward(self, inp, weights, reuse=False, scope='', stop_grad=False, stop_at_grad=False, noise=True,
                action_label=None):
        weights = weights.copy()
        batch_size = tf.shape(inp)[0]

        def swish(inp):
            return inp * tf.nn.sigmoid(inp)

        joint = inp
        joint = tf.reshape(joint, (-1, FLAGS.input_objects * self.dim_input))

        if action_label is not None:
            joint = tf.concat([joint, action_label], axis=1)

        h1 = tf.nn.relu(tf.matmul(joint, weights['ff_w1']) + weights['ff_b1'])
        h2 = tf.nn.relu(tf.matmul(h1, weights['ff_w2']) + weights['ff_b2'])
        h3 = (tf.matmul(h2, weights['ff_w3']) + weights['ff_b3'])
        energy = tf.matmul(h3, weights['ff_w6'])

        return energy
