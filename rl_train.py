import gym
import numpy as np
import tensorflow as tf
# from tensorflow.nn.rnn_cell import LSTMCell
import os.path as osp
import os
from envs import Point, Maze
from utils import ReplayBuffer, parse_valid_obs

from baselines.logger import TensorBoardOutputFormat
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from tensorflow.python.platform import flags
from traj_model import TrajNetLatentFC, TrajInverseDynamics, TrajFFDynamics
from custom_adam import AdamOptimizer
from baselines.bench import Monitor
import seaborn as sns
import  matplotlib.pyplot as plt

FLAGS = flags.FLAGS

flags.DEFINE_string('datasource', 'point', 'point or maze')
flags.DEFINE_string('exp', 'rl_default', 'name of experiment')
flags.DEFINE_integer('num_env', 128, 'batch size and number of planning steps')
flags.DEFINE_string('logdir', 'cachedir', 'location where log of rl experiments will be stored')
flags.DEFINE_string('model', 'ebm', 'ebm or ff')
flags.DEFINE_bool('train', True, 'whether to train with environmental interaction or not')
flags.DEFINE_bool('debug', False, 'print out outputs of information')
flags.DEFINE_integer('save_interval', 1000, 'save outputs every so many batches')
flags.DEFINE_integer('test_interval', 1000, 'evaluate outputs every so many batches')
flags.DEFINE_integer('log_interval', 10, 'interval to log values')
flags.DEFINE_integer('resume_iter', -1, 'iteration to resume training from')
flags.DEFINE_float('lr', 1e-3, 'Learning for training')
flags.DEFINE_integer('seed', 0, 'Value of seed')
flags.DEFINE_bool('heatmap', False, 'Visualize the heatmap in environments')

# Architecture Settings
flags.DEFINE_integer('num_filters', 64, 'number of filters for networks')
flags.DEFINE_integer('latent_dim', 24, 'Number of dimension encoding state of object')
flags.DEFINE_integer('action_dim', 24, 'Number of dimension for encoding action of object')
flags.DEFINE_integer('input_objects', 1, 'Number of objects to predict the trajectory of.')
flags.DEFINE_bool('spec_norm', True, 'Whether to use spectral normalization on weights')
flags.DEFINE_bool('fast_goal', False, 'Try to reach the goal quicly')
flags.DEFINE_bool('v_penalty', False, 'Penalty for distance between two points')

# EBM settings
flags.DEFINE_integer('num_steps', 20, 'Steps of gradient descent for training')
flags.DEFINE_integer('total_frame', 2, 'Number of frames to use')
flags.DEFINE_bool('replay_batch', True, 'Whether to use a replay buffer for samples')
flags.DEFINE_bool('cond', False, 'Whether to condition on actions')
flags.DEFINE_integer('temperature', 1, 'Temperature for energy function')
flags.DEFINE_bool('inverse_dynamics', True, 'Whether to train a inverse dynamics model')
flags.DEFINE_bool('gt_inverse_dynamics', True, 'Whether to train a inverse dynamics model')
flags.DEFINE_integer('num_plan_steps', 50, 'Steps of planning')

# Settings for MCMC
flags.DEFINE_float('step_lr', 1.0, 'Size of steps for gradient descent')
flags.DEFINE_integer('plan_steps', 20, 'Number of steps of planning')
flags.DEFINE_bool('anneal', False, 'To anneal sampling')

# Environment Interaction Settings
flags.DEFINE_float('nsteps', 1e6, 'Number of steps of environment interaction')

if FLAGS.datasource == "maze" or FLAGS.datasource == "point":
    FLAGS.latent_dim = 2
    FLAGS.action_dim = 2

def safemean(xs):
    return np.nan if len(xs) == 0 else np.mean(xs)

def train(target_vars, saver, sess, logger, resume_iter, env):
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
    loss_total = target_vars['loss_total']
    dyn_loss = target_vars['dyn_loss']
    dyn_dist = target_vars['dyn_dist']

    ob = env.reset()[:, None, None, :]

    output = [train_op, x_mod]
    log_output = [train_op, dyn_loss, dyn_dist, energy_pos, energy_neg, loss_ml, loss_total, x_grad, action_grad, x_mod]

    print(log_output)
    replay_buffer = ReplayBuffer(1000000)
    pos_replay_buffer = ReplayBuffer(1000000)

    epinfos = []
    points = []
    total_obs = []
    for itr in range(resume_iter, tot_iter):
        x_plan = np.random.uniform(-1, 1, (FLAGS.num_env, FLAGS.plan_steps, 1, 2))
        action_plan = np.random.uniform(-1, 1, (FLAGS.num_env, FLAGS.plan_steps + 1, 2))
        if FLAGS.datasource == "maze":
            x_end = np.tile(np.array([[0.7, -0.8]]), (FLAGS.num_env, 1))[:, None, None, :]
        else:
            x_end = np.tile(np.array([[0.5, 0.5]]), (FLAGS.num_env, 1))[:, None, None, :]
        x_traj, traj_actions = sess.run([x_joint, actions], {X_START: ob, X_PLAN: x_plan, X_END: x_end, ACTION_PLAN: action_plan})

        # Add some amount of exploration into predicted actions
        # traj_actions = traj_actions + np.random.uniform(-0.1, 0.1, traj_actions.shape)
        traj_actions = np.clip(traj_actions, -1, 1)

        if FLAGS.debug:
            print(x_traj[0])

        obs = [ob[:, 0, 0, :]]
        dones = []
        diffs = []
        for i in range(traj_actions.shape[1] - 1):
            action = traj_actions[:, i]
            # action = np.random.uniform(-1, 1, traj_actions[:, i].shape)
            ob, _, done, infos = env.step(action)

            if i == 0:
                print(x_traj[0, 0], x_traj[0, 1], ob[0])
                print(x_traj[0, :].max(), x_traj[0:, 1].min())


            dones.append(done)
            obs.append(ob)

            for info in infos:
                maybeepinfo = info.get('episode')
                if maybeepinfo: epinfos.append(maybeepinfo)

            diffs.append(np.abs(x_traj[:, i+1] - ob).mean())

        ob =  ob[:, None, None, :]
        dones = np.array(dones).transpose()
        obs = np.stack(obs, axis=1)[:, :, None, :]

        if FLAGS.heatmap:
            total_obs.append(obs.reshape((-1, FLAGS.latent_dim)))

        action, ob_pair = parse_valid_obs(obs, traj_actions, dones)

        x_noise = np.stack([x_traj[:, :-1], x_traj[:, 1:]], axis=2)
        s = x_noise.shape
        x_noise_neg = x_noise.reshape((s[0] * s[1], s[2], s[3], s[4]))
        action_noise_neg = traj_actions[:, 1:]
        s = action_noise_neg.shape
        action_noise_neg = action_noise_neg.reshape((s[0]*s[1], s[2]))

        traj_action_encode = action.reshape((-1, 1, 1, FLAGS.action_dim))
        encode_data = np.concatenate([ob_pair, np.tile(traj_action_encode, (1, FLAGS.total_frame, 1, 1))], axis=3)
        pos_replay_buffer.add(encode_data)

        if len(pos_replay_buffer) > FLAGS.num_env * FLAGS.plan_steps:
            sample_data = pos_replay_buffer.sample(FLAGS.num_env * FLAGS.plan_steps)
            sample_ob = sample_data[:, :, :, :-FLAGS.action_dim]
            sample_actions = sample_data[:, 0, 0, -FLAGS.action_dim:]

            ob_pair = np.concatenate([ob_pair, sample_ob], axis=0)
            action = np.concatenate([action, sample_actions], axis=0)


        feed_dict = {X: ob_pair, X_NOISE: x_noise_neg, ACTION_NOISE: action_noise_neg, ACTION_LABEL: action}

        batch_size = x_noise_neg.shape[0]
        if FLAGS.replay_batch and len(replay_buffer) > batch_size:
            replay_batch = replay_buffer.sample(batch_size)
            replay_mask = (np.random.uniform(0, 1, (batch_size)) > 0.2)
            feed_dict[X_NOISE][replay_mask] = replay_batch[replay_mask]


        if itr % FLAGS.log_interval == 0:
            _, dyn_loss, dyn_dist, e_pos, e_neg, loss_ml, loss_total, x_grad, action_grad, x_mod = sess.run(log_output, feed_dict=feed_dict)
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
            kvs["diffs"] = diffs[-1]

            epinfos = []

            string = "Obtained a total of "
            for key, value in kvs.items():
                string += "{}: {}, ".format(key, value)

            print(string)
            logger.writekvs(kvs)
        else:
            _, x_mod = sess.run(output, feed_dict=feed_dict)

        if FLAGS.replay_batch and (x_mod is not None):
            replay_buffer.add(x_mod)

        if itr % FLAGS.save_interval == 0:
            saver.save(sess, osp.join(FLAGS.logdir, FLAGS.exp, 'model_{}'.format(itr)))

        if FLAGS.heatmap and itr == 200:
            total_obs = np.concatenate(total_obs, axis=0)
            total_obs = total_obs[np.random.permutation(total_obs.shape[0])[:2000000]]
            sns.jointplot(x=total_obs[:, 0], y=total_obs[:, 1], kind="kde")
            plt.savefig("kde.png")
            assert False


def construct_ff_plan_model(model, weights, X_PLAN, X_START, X_END, ACTION_PLAN, target_vars={}):
    actions = ACTION_PLAN
    steps = tf.constant(0)
    c = lambda i, x, y: tf.less(i, FLAGS.num_plan_steps)

    def mcmc_step(counter, actions, x_vals):
        actions = actions + tf.random_normal(tf.shape(actions), mean=0.0, stddev=0.01)

        x_val = X_START
        x_vals = []
        for i in range(FLAGS.plan_steps + 1):
            x_val = model.forward(x_val, weights, action_label=actions[:, i])
            x_vals.append(x_val)

        x_vals = tf.stack(x_vals, axis=1)

        if FLAGS.anneal:
            anneal_val = tf.cast(counter, tf.float32) / FLAGS.num_steps
        else:
            anneal_val = 1

        energy = tf.reduce_sum(tf.square(x_val - X_END[:, 0, 0]))

        action_grad = tf.gradients(energy, [actions])[0]
        actions = actions - FLAGS.step_lr * anneal_val * action_grad
        actions = tf.clip_by_value(actions, -1.0, 1.0)

        counter = counter + 1

        return counter, actions, x_vals

    steps, actions, x_joint = tf.while_loop(c, mcmc_step, (steps, actions, tf.concat([X_START, X_PLAN], axis=1)[:, :, 0]))

    target_vars['x_joint'] = tf.expand_dims(x_joint, axis=2)
    target_vars['actions'] = actions
    target_vars['X_START'] = X_START
    target_vars['X_END'] = X_END
    target_vars['X_PLAN'] = X_PLAN
    target_vars['ACTION_PLAN'] = ACTION_PLAN

    return target_vars

def construct_plan_model(model, weights, X_PLAN, X_START, X_END, ACTION_PLAN, target_vars={}):
    actions = ACTION_PLAN
    x_joint = tf.concat([X_START, X_PLAN], axis=1)
    steps = tf.constant(0)
    c = lambda i, x, y: tf.less(i, FLAGS.num_plan_steps)

    def mcmc_step(counter, x_joint, actions):
        if FLAGS.cond:
            actions = actions + tf.random_normal(tf.shape(actions), mean=0.0, stddev=0.01)
        x_joint = x_joint + tf.random_normal(tf.shape(x_joint), mean=0.0, stddev=0.01)
        cum_energies = 0
        for i in range(FLAGS.plan_steps - FLAGS.total_frame + 2):
            cum_energy = model.forward(x_joint[:, i:i + FLAGS.total_frame], weights, action_label=actions[:, i])
            cum_energies = cum_energies + cum_energy

        if FLAGS.anneal:
            anneal_val = tf.cast(counter, tf.float32) / FLAGS.num_steps
        else:
            anneal_val = 1

        if FLAGS.debug:
            cum_energies = tf.Print(cum_energies, [cum_energies], message="energy")

        if FLAGS.fast_goal:
            cum_energies = cum_energies + tf.reduce_sum(tf.square(x_joint - X_END))

        if FLAGS.v_penalty:
            cum_energies = cum_energies + 0.001 * tf.reduce_mean(tf.square(x_joint[:, 1:] - x_joint[:, :-1]))

        # cum_energies = cum_energies + 1e-5 * tf.reduce_sum(tf.abs(x_joint[:, -1:] - X_END))

        x_grad, action_grad = tf.gradients(cum_energies, [x_joint, actions])
        x_joint = x_joint - FLAGS.step_lr  *  anneal_val * x_grad
        x_joint = tf.concat([X_START, x_joint[:, 1:FLAGS.plan_steps+1]], axis=1)
        x_joint = tf.clip_by_value(x_joint, -1.0, 1.0)

        if FLAGS.cond:
            actions = actions - FLAGS.step_lr * anneal_val * action_grad
            actions = tf.clip_by_value(actions, -1.0, 1.0)

        counter = counter + 1

        return counter, x_joint, actions

    steps, x_joint, actions = tf.while_loop(c, mcmc_step, (steps, x_joint, actions))

    if FLAGS.gt_inverse_dynamics:
        actions = tf.clip_by_value(20.0 * (x_joint[:, 1:, 0] - x_joint[:, :-1, 0]), -1.0, 1.0)

    target_vars['x_joint'] = x_joint
    target_vars['actions'] = actions
    target_vars['X_START'] = X_START
    target_vars['X_END'] = X_END
    target_vars['X_PLAN'] = X_PLAN
    target_vars['ACTION_PLAN'] = ACTION_PLAN

    return target_vars


def construct_ff_model(model, weights, X_NOISE, X, ACTION_LABEL, ACTION_NOISE_LABEL, optimizer):
    target_vars = {}
    x_mods = []

    x_pred = model.forward(X[:, 0, 0], weights, action_label=ACTION_LABEL)
    loss_total = tf.reduce_mean(tf.square(x_pred - X[:, 1, 0]))
    target_vars['dyn_loss'] = tf.zeros(1)
    target_vars['dyn_dist'] = tf.zeros(1)

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

    target_vars['loss_ml'] = tf.zeros(1)
    target_vars['loss_total'] = loss_total
    target_vars['gvs'] = gvs
    target_vars['loss_energy'] = tf.zeros(1)

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

    return target_vars


def construct_model(model, weights, X_NOISE, X, ACTION_LABEL, ACTION_NOISE_LABEL, optimizer):
    target_vars = {}
    x_mods = []

    energy_pos = model.forward(X, weights, action_label=ACTION_LABEL)
    energy_noise = energy_start = model.forward(X_NOISE, weights, reuse=True, stop_at_grad=True, action_label=ACTION_LABEL)

    x_mod = X_NOISE

    x_grads = []
    x_ees = []

    if FLAGS.inverse_dynamics:
        dyn_model = TrajInverseDynamics()
        weights = dyn_model.construct_weights(scope="inverse_dynamics", weights=weights)

    steps = tf.constant(0)
    c = lambda i, x, y: tf.less(i, FLAGS.num_steps)

    def mcmc_step(counter, x_mod, action_label):
        x_mod = x_mod + tf.random_normal(tf.shape(x_mod), mean=0.0, stddev=0.01)
        action_label = action_label + tf.random_normal(tf.shape(action_label), mean=0.0, stddev=0.01)
        energy_noise = model.forward(x_mod, weights, action_label=action_label, reuse=True)
        lr =  FLAGS.step_lr

        x_grad = tf.gradients(FLAGS.temperature * energy_noise, [x_mod])[0]

        x_mod = x_mod - lr * x_grad

        if FLAGS.cond:
            x_grad, action_grad = tf.gradients(FLAGS.temperature * energy_noise, [x_mod, action_label])
        else:
            x_grad, action_grad = tf.gradients(FLAGS.temperature * energy_noise, [x_mod])[0], tf.zeros(1)

        action_label = action_label - FLAGS.step_lr * action_grad

        x_mod = tf.clip_by_value(x_mod, -1.2, 1.2)
        action_label = tf.clip_by_value(action_label, -1.2, 1.2)

        counter = counter + 1

        return counter, x_mod, action_label


    steps, x_mod, action_label = tf.while_loop(c, mcmc_step, (steps, x_mod, ACTION_NOISE_LABEL))

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
        pos_loss = tf.reduce_mean(temp * energy_pos)
        neg_loss = -tf.reduce_mean(temp * energy_neg)
        loss_ml = (pos_loss + tf.reduce_sum(neg_loss))
        loss_total = tf.reduce_mean(loss_ml)
        loss_total = loss_total + \
            (tf.reduce_mean(tf.square(energy_pos)) + tf.reduce_mean(tf.square((energy_neg))))

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

        if FLAGS.inverse_dynamics:
            train_op = tf.group(train_op, dyn_train_op)

        target_vars['train_op'] = train_op

        print("Finished applying gradients.")
        target_vars['loss_ml'] = loss_ml
        target_vars['loss_total'] = loss_total
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
    target_vars['ACTION_LABEL'] = ACTION_LABEL
    target_vars['ACTION_NOISE_LABEL'] = ACTION_NOISE_LABEL

    return target_vars


def main():
    logdir = osp.join(FLAGS.logdir, FLAGS.exp)
    if not osp.exists(logdir):
        os.makedirs(logdir)
    logger = TensorBoardOutputFormat(logdir)

    datasource = FLAGS.datasource
    def make_env(rank):
        def _thunk():

            # Make the environments non stoppable for now
            if datasource == "maze":
                env = Maze(end=[0.7, -0.8], start=[-0.85, -0.85], random_starts=False)
            elif datasource == "point":
                env = Point(end=[0.5, 0.5], start=[0.0, 0.0], random_starts=True)
            env.seed(rank)
            env = Monitor(env, os.path.join("/tmp", str(rank)), allow_early_resets=True)
            return env

        return _thunk

    env = SubprocVecEnv([make_env(i + FLAGS.seed) for i in range(FLAGS.num_env)])

    if FLAGS.datasource == 'point' or FLAGS.datasource == 'maze':
        if FLAGS.model == "ebm":
            model = TrajNetLatentFC(dim_input=FLAGS.total_frame)
        elif FLAGS.model == "ff":
            model = TrajFFDynamics(dim_input=FLAGS.latent_dim, dim_output=FLAGS.latent_dim)
        X_NOISE = tf.placeholder(shape=(None, FLAGS.total_frame, FLAGS.input_objects, FLAGS.latent_dim), dtype=tf.float32)
        X = tf.placeholder(shape=(None, FLAGS.total_frame, FLAGS.input_objects, FLAGS.latent_dim), dtype = tf.float32)

        ACTION_LABEL = tf.placeholder(shape=(None, 2), dtype=tf.float32)
        ACTION_NOISE_LABEL = tf.placeholder(shape=(None, 2), dtype=tf.float32)
        ACTION_PLAN = tf.placeholder(shape=(None, FLAGS.plan_steps+1, 2), dtype=tf.float32)

        X_START = tf.placeholder(shape=(None, 1, FLAGS.input_objects, FLAGS.latent_dim), dtype = tf.float32)
        X_PLAN = tf.placeholder(shape=(None, FLAGS.plan_steps, FLAGS.input_objects, FLAGS.latent_dim), dtype = tf.float32)
        X_END = tf.placeholder(shape=(None, 1, FLAGS.input_objects, FLAGS.latent_dim), dtype = tf.float32)

    weights = model.construct_weights(action_size=FLAGS.action_dim)
    optimizer = AdamOptimizer(1e-2, beta1=0.0, beta2=0.999)

    if FLAGS.model == "ff":
        target_vars = construct_ff_model(model, weights, X_NOISE, X, ACTION_LABEL, ACTION_NOISE_LABEL, optimizer)
        target_vars = construct_ff_plan_model(model, weights, X_PLAN, X_START, X_END, ACTION_PLAN, target_vars=target_vars)
    else:
        target_vars = construct_model(model, weights, X_NOISE, X, ACTION_LABEL, ACTION_NOISE_LABEL, optimizer)
        target_vars = construct_plan_model(model, weights, X_PLAN, X_START, X_END, ACTION_PLAN, target_vars=target_vars)

    sess = tf.InteractiveSession()
    saver = loader = tf.train.Saver(max_to_keep=10, keep_checkpoint_every_n_hours=2)

    tf.global_variables_initializer().run()
    print("Initializing variables...")

    if FLAGS.resume_iter != -1 or not FLAGS.train:
        model_file = osp.join(logdir, 'model_{}'.format(FLAGS.resume_iter))
        resume_itr = FLAGS.resume_iter
        saver.restore(sess, model_file)

    train(target_vars, saver, sess, logger, FLAGS.resume_iter, env)

if __name__ == "__main__":
    main()
