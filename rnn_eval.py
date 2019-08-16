import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorflow.python.platform import flags
import os.path as osp
import os
from rl_algs.logger import TensorBoardOutputFormat
from itertools import product
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import random

FLAGS = flags.FLAGS
flags.DEFINE_integer('batch_size', 256, 'Size of inputs')
flags.DEFINE_integer('output_dim', 48, 'Size of prediction')
flags.DEFINE_integer('input_dim', 48, 'Size of prediction')
flags.DEFINE_integer('n_epoch', 10, 'Number of epochs to train')
flags.DEFINE_integer('plan_steps', 30, 'Number of steps to predict from')
flags.DEFINE_bool('train', True, 'By default train model')



class SimpleNet(nn.Module):

    def __init__(self, flags):
        super(SimpleNet, self).__init__()

        self.fc1 = nn.Linear(48, 128)
        self.lstm = torch.nn.LSTM(128, 128)
        self.fc2 = nn.Linear(128, 48)

    def forward(self, inp):
        encode = F.relu(self.fc1(inp))
        output, state = self.lstm(encode)
        output = self.fc2(output)

        return output


def main():
    data = np.load("data/collision.npz")
    dataset = data['arr_0']
    mask = data['arr_1']

    s = dataset.shape
    dataset  = dataset.reshape((*s[:-2], 48))

    split_idx = int(dataset.shape[0] * 0.9)
    dataset_train = dataset_orig_train = dataset[:split_idx]
    mask_train = mask[:split_idx]
    s = mask_train.shape
    mask_train = np.tile(mask_train[:, :, :, None], (1, 1, 1, 6))
    mask_train = mask_train.reshape((*s[:-1], 48))

    print("train shapes ", mask_train.shape, dataset_train.shape)

    dataset_train = dataset_train * (1 - mask_train[:, :, :])

    dataset_test = dataset_test_orig = dataset[split_idx:]
    mask_test = mask[split_idx:]
    s = mask_test.shape
    mask_test = np.tile(mask_test[:, :, :, None], (1, 1, 1, 6))
    mask_test = mask_test.reshape((*s[:-1], 48))
    dataset_test = dataset_test * (1 - mask_test[:, :, :])

    model = SimpleNet(FLAGS).cuda()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    if FLAGS.train:
        random_combo = list(product(range(0, dataset_train.shape[1]-FLAGS.plan_steps-1, FLAGS.plan_steps),
                                    range(0, dataset_train.shape[0] - FLAGS.batch_size, FLAGS.batch_size)))

        for epoch in tqdm(range(FLAGS.n_epoch)):
            for j, i in random_combo:
                optimizer.zero_grad()

                inp_dat = dataset_train[i:i+FLAGS.batch_size, j:j+FLAGS.plan_steps]
                inp_dat = inp_dat.transpose((1, 0, 2))

                label_dat = dataset_train[i:i+FLAGS.batch_size, j+1:j+FLAGS.plan_steps+1]
                label_dat = label_dat.transpose((1, 0, 2))

                inp_dat, label_dat = torch.from_numpy(inp_dat).float().cuda(), torch.from_numpy(label_dat).float().cuda()
                # print("mask_train shape ", mask_train.shape)
                mask_select = mask_train[i:i+FLAGS.batch_size, j+1:j+FLAGS.plan_steps+1]
                mask_postprocess = torch.from_numpy(mask_select.transpose((1, 0, 2))).float().cuda()

                output = model.forward(inp_dat)
                # print(output.size(), mask_postprocess.size(), label_dat.size())
                loss = (output * (1 - mask_postprocess) - label_dat).pow(2).mean()

                loss.backward()
                optimizer.step()


        torch.save(model.state_dict(), "physics")
    else:
        torch.load("physics")


    # Script for testing
    errors = []
    for i in tqdm(range(10)):
        batch = random.randint(0, dataset_test.shape[0] - FLAGS.batch_size)
        it = random.randint(0, dataset_test.shape[1] - FLAGS.plan_steps - 1)

        masked_traj = dataset_test[batch:batch+FLAGS.batch_size, it:it+FLAGS.plan_steps]
        true_unmask = dataset_test_orig[batch:batch+FLAGS.batch_size, it+1:it+FLAGS.plan_steps+1]

        mask = mask_test[batch:batch+FLAGS.batch_size, it:it+FLAGS.plan_steps, :, None]

        masked_traj = masked_traj.transpose((1, 0, 2))
        true_unmask = true_unmask.transpose((1, 0, 2))

        masked_traj, true_unmask = torch.from_numpy(masked_traj).float().cuda(), torch.from_numpy(true_unmask).float().cuda()

        output = model.forward(masked_traj)

        error = torch.abs(true_unmask - output).mean().item()

        errors.append(error)


    print("Obtained an error of ", np.mean(errors))



if __name__ == "__main__":
    main()
