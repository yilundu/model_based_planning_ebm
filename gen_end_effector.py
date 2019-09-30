import matplotlib.pyplot as plt
import os.path as osp
import os
import tensorflow as tf
import seaborn as sns
import numpy as np

sns.set(font_scale=2.0)

base_path = "cachedir"
random_action_exp = "reacher_704_random_action"
# directed_exp = "reacher_704_optimize"
# directed_exps = ["reacher_704_explore_2",  "818_end_effector_stat_1", "818_end_effector_stat_2", "818_end_effector_stat_3"]
directed_exps = ["818_end_effector_stat_1", "818_end_effector_stat_2", "818_end_effector_stat_3"]



def load_exp(name):
    name = osp.join(base_path, name)
    events = list(filter(lambda x: 'events' in x, os.listdir(name)))
    events_path = [osp.join(name, event) for event in events]
    assert (len(events_path) == 1)

    logs = [[e for e in tf.train.summary_iterator(event_path)] for event_path in events_path]
    log = logs[0]

    end_effector_occupancy = [sum([tup.simple_value for tup in t.summary.value if tup.tag == "end_effector_occupancy"])  for t in log]
    num_env_steps = [sum([tup.simple_value for tup in t.summary.value if tup.tag == "num_env_steps"])  for t in log]

    return np.array(end_effector_occupancy), np.array(num_env_steps)


def collect_stats(path):
    random_effector, random_env_steps = load_exp(random_action_exp)
    max_occupancy = random_effector.max() + 5
    random_ax, = plt.semilogx(random_env_steps, random_effector / max_occupancy * 100, 'r')
    max_step = None

    occupancies = []
    directed_effectors = []
    for i, directed_exp in enumerate(directed_exps):
        directed_effector, directed_env_steps = load_exp(directed_exp)
        ix = np.searchsorted(directed_env_steps, 8000)
        random_effector = random_effector / max_occupancy * 100.
        directed_effector = directed_effector / max_occupancy * 100.
        directed_ax, = plt.semilogx(directed_env_steps[:ix], directed_effector[:ix], 'b')
        directed_effectors.append(directed_effector[:ix])

    plt.xlabel("Environment Steps (Log)")
    plt.ylabel("Occupancy (%)")
    plt.title("Finger 3D Occupancy")
    plt.xscale("log")

    directed_max_steps = directed_env_steps.max()
    print("occupancy mean, std ", np.mean(occupancies), np.std(occupancies))
    random_idx = np.searchsorted(random_env_steps, 8000)
    print("random value ", random_effector[random_idx-1])

    # plt.xlim(0, directed_max_steps)
    plt.legend((random_ax, directed_ax), ('Random', 'EBM'))
    plt.tight_layout()
    plt.savefig("end_effector_explore.pdf")



if __name__ == "__main__":
    path = "reacher.png"
    collect_stats(path)
