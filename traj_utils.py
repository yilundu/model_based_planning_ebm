from trajopt.envs.continual_reacher_env import ContinualReacher7DOFEnv
from multiprocessing import Pool
import numpy as np



def set_env(env, ob):
    sim = env.sim

    sim.data.qpos[:] = ob[0:7]
    sim.data.qvel[:] = ob[7:14] * 10
    sim.data.site_xpos[env.hand_sid] = ob[14:17]
    env.model.site_pos[env.target_sid] = ob[17:20]

    sim.forward()


def get_score(ob, goal):
    score = -np.square(ob - goal).sum()
    return score


def process_trajectory(traj, large=False):
    env = ContinualReacher7DOFEnv()
    env.reset()
    sim = env.sim

    actions = []
    eps_random = 0.1

    mean = np.zeros(env.action_dim)

    if large:
        sigma = 0.15
        n_sim = 100
        n_iter = 40
        kappa = 25
    else:
        sigma = 0.20
        n_sim = 10
        n_iter = 10
        kappa = 25


    for i in range(traj.shape[0]-1):
        random_action = np.random.uniform(-0.3, 0.3, (7,))
        for j in range(n_iter):
            perturb_action = np.random.normal(0, sigma, size=(n_sim, 7))
            scores = []
            proposal_actions = []
            for k in range(n_sim):
                try:
                    propose_action = random_action + perturb_action[k]
                    propose_action = np.clip(propose_action, -1, 1)
                    set_env(env, traj[i])
                    ob, _, _, _ = env.step(propose_action)
                    score = get_score(ob, traj[i+1])

                    proposal_actions.append(propose_action)
                    scores.append(score)
                except:
                    continue

            scores = np.array(scores)
            proposal_actions = np.array(proposal_actions)

            exp_scores = np.exp(kappa * (scores[:, None] - scores[:, None].max()))
            softmax = exp_scores / (exp_scores.sum() + 1e-10)

            random_action = (softmax * proposal_actions).sum(axis=0)

        actions.append(random_action)

    return np.array(actions)

def continual_reacher_inverse_dynamics(x_plan):
    inverse_pool = Pool()
    x_plan = list(x_plan)
    actions = inverse_pool.map(process_trajectory, x_plan)
    inverse_pool.close()
    return np.array(actions)


def linear_reacher_inverse_dynamics(x_traj, action, state_matrix):
    actions = []

    for i in range(x_traj.shape[1]-1):
        action = linear_reacher_helper(x_traj[:, i, :-3], x_traj[:, i+1, :-3], action, state_matrix)
        actions.append(action)

    actions = np.stack(actions, axis=1)
    return actions


def linear_reacher_helper(s1, s2, a_prev, state_matrix):

    # print(s1.shape)
    # print(s2.shape)
    # print(a_prev.shape)
    # print(state_matrix.shape)
    lam = 0.3
    # Generate target of the form dxn
    target = (s2.transpose() - state_matrix[:, :17].dot(s1.transpose())).squeeze()
    target_2 = lam * a_prev.transpose().squeeze()

    # Generate multiplication matrix of from dxi where i is dimension of action
    m1 = state_matrix[:, 17:]
    m2 = lam * np.eye(7)

    # print("target shape ", target.shape, target_2.shape)
    target = np.concatenate([target, target_2], axis=0)
    m = np.concatenate([m1, m2], axis=0)
    controls, _, _, _ = np.linalg.lstsq(m, target)
    # print(m)
    # print(target)
    # print(controls.shape)
    # print(controls)
    # assert False

    controls = controls.transpose()

    return controls


def update_linear_weight(s1, s2, a, state_matrix, alpha):
    s1 = s1[:, :-3]
    s2 = s2[:, :-3]

    concat_state = np.concatenate([s1, a], axis=1)
    col = (s1 - state_matrix.dot(concat_state.transpose()).transpose()) / (np.linalg.norm(concat_state, axis=1, keepdims=True) ** 2 + alpha)

    state_update = (col[:, :, None] * concat_state[:, None, :]).mean(axis=0)

    return state_matrix + state_update


if __name__ == "__main__":
    dat = np.load("debug.npz")
    obs, gt_actions = dat['traj'], dat['actions']
    actions = process_trajectory(obs)

    # print("predicted actions: ", actions)
    # print("gt actions: ", gt_actions)

    print("average_dist", np.abs(gt_actions - actions).mean())

    diffs = []
    env = ContinualReacher7DOFEnv()
    for i in range(obs.shape[0] - 1):
        set_env(env, obs[i])
        ob, _, _, _ = env.step(actions[i])
        diffs.append(np.abs(obs[i+1] - ob).mean())
        # print(ob, obs[i+1])

    print("average difference ", np.mean(diffs))
