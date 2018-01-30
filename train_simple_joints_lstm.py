import shutil

import numpy as np
from torch import autograd, nn, optim, torch
from torch.utils.data import DataLoader

# absolute imports here, so that you can run the file directly
from simple_joints_lstm.lstm_simple_net import LstmSimpleNet
from simple_joints_lstm.mujoco_simple1_dataset import MujocoSimple1Dataset
from simple_joints_lstm.params import *

dataset = MujocoSimple1Dataset(DATASET_PATH)

# batch size has to be 1, otherwise the LSTM doesn't know what to do
dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=1)

net = LstmSimpleNet()

if CUDA:
    net.cuda()


def makeIntoVariables(dataslice):
    x, y = autograd.Variable(
        dataslice["state_next_sim_joints"],
        requires_grad=False
    ), autograd.Variable(
        dataslice["state_next_real_joints"],
        requires_grad=False
    )
    if CUDA:
        return x.cuda()[0], y.cuda()[0]
    return x[0], y[0]  # because we have minibatch_size=1


def printEpisodeLoss(epoch_idx, episode_idx, loss_episode, diff_episode, len_episode):
    print("epoch {}, episode {}, "
          "loss: {}, loss avg: {}, "
          "diff: {}, diff avg: {}".format(
        epoch_idx,
        episode_idx,
        round(loss_episode, 2),
        round(float(loss_episode) / len_episode, 2),
        round(diff_episode, 2),
        round(float(diff_episode) / len_episode, 2)
    ))


def printEpochLoss(epoch_idx, episode_idx, loss_epoch, diff_epoch):
    print("epoch {}, "
          "loss: {}, loss avg: {}, "
          "diff: {}, diff avg: {}".format(
        epoch_idx,
        round(loss_epoch, 2),
        round(float(loss_epoch) / (episode_idx + 1), 2),
        round(diff_epoch, 2),
        round(float(diff_epoch) / (episode_idx + 1), 2)
    ))


def saveModel(state, epoch, epoch_loss, epoch_diff, is_best):
    torch.save({
        "epoch": epoch,
        "state_dict": state,
        "epoch_avg_loss": epoch_loss,
        "epoch_avg_diff": epoch_diff
    }, MODEL_PATH)
    if is_best:
        shutil.copyfile(MODEL_PATH, MODEL_PATH_BEST)


loss_function = nn.MSELoss()
optimizer = optim.Adam(net.parameters())

loss_history = [9999999] # very high loss because loss can't be empty for min()

for epoch_idx in np.arange(EPOCHS):

    loss_epoch = 0
    diff_epoch = 0

    for episode_idx, data in enumerate(dataloader):
        x, y = makeIntoVariables(data)

        # reset hidden lstm units
        net.hidden = net.init_hidden()

        loss_episode = 0
        optimizer.zero_grad()

        # iterate over episode frames
        for frame_idx in np.arange(len(x)):
            # x_frame = x[frame_idx]
            # y_frame = y[frame_idx]

            prediction = net.forward(x[frame_idx])
            loss = loss_function(prediction, y[frame_idx].view(1, -1))

            loss_episode += loss.data.cpu()[0]
            loss.backward(retain_variables=True)

        optimizer.step()

        loss.detach()
        net.hidden[0].detach()
        net.hidden[1].detach()

        diff_episode = torch.sum(torch.pow(x.data - y.data, 2))
        printEpisodeLoss(epoch_idx, episode_idx, loss_episode, diff_episode, len(x))

        loss_epoch += loss_episode
        diff_epoch += diff_episode

    printEpochLoss(epoch_idx, episode_idx, loss_epoch, diff_epoch)
    saveModel(
        state=net.state_dict(),
        epoch=epoch_idx,
        epoch_loss=loss_epoch,
        epoch_diff=diff_epoch,
        is_best=(loss_epoch<min(loss_history))
    )
    loss_history.append(loss_epoch)
