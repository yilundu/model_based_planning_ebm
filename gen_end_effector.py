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
directed_exp = "reacher_704_explore_2"


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
    directed_effector, directed_env_steps = load_exp(directed_exp)

    random_ax, = plt.plot(random_env_steps, random_effector)
    directed_ax, = plt.plot(directed_env_steps, directed_effector)
    directed_max_steps = directed_env_steps.max()
    plt.xlabel("Environment Steps")
    plt.ylabel("Occupancy")
    plt.title("Finger 3D Occupancy")

    plt.xlim(0, directed_max_steps)
    plt.legend((random_ax, directed_ax), ('Random', 'EBM'))
    plt.tight_layout()
    plt.savefig("end_effector_explore.pdf")



if __name__ == "__main__":
    path = "reacher.png"
    collect_stats(path)
