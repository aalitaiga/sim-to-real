import shutil

import numpy as np
from torch import autograd, nn, optim, torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

# absolute imports here, so that you can run the file directly
from simple_joints_lstm.mujoco_traintest_dataset_cheetah import MujocoTraintestCheetahDataset
from simple_joints_lstm.params_cheetah import *
from simple_joints_lstm.lstm_simple_net2_cheetah import LstmSimpleNet2Cheetah
import os

try:
    from hyperdash import Experiment

    hyperdash_support = True
except:
    hyperdash_support = False

dataset = MujocoTraintestCheetahDataset(DATASET_PATH, for_training=TRAIN)

# batch size has to be 1, otherwise the LSTM doesn't know what to do
dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=1)

net = LstmSimpleNet2Cheetah()

print(net)

if CUDA:
    net.cuda()

if TRAIN:
    print("STARTING IN TRAINING MODE")
else:
    print("STARTING IN VALIDATION MODE")


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
    loss_avg = round(float(loss_episode) / len_episode, 2)
    diff_avg = round(float(diff_episode) / len_episode, 2)
    print("epoch {}, episode {}, "
          "loss: {}, loss avg: {}, "
          "diff: {}, diff avg: {}".format(
        epoch_idx,
        episode_idx,
        round(loss_episode, 2),
        loss_avg,
        round(diff_episode, 2),
        diff_avg
    ))
    if hyperdash_support:
        exp.metric("diff avg", diff_avg)
        exp.metric("loss avg", loss_avg)


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


def saveModel(state, epoch, loss_epoch, diff_epoch, is_best, episode_idx):
    torch.save({
        "epoch": epoch,
        "episodes": episode_idx + 1,
        "state_dict": state,
        "epoch_avg_loss": float(loss_epoch) / (episode_idx + 1),
        "epoch_avg_diff": float(diff_epoch) / (episode_idx + 1)
    }, MODEL_PATH)
    if is_best:
        shutil.copyfile(MODEL_PATH, MODEL_PATH_BEST)


def loadModel(optional=True):
    model_exists = os.path.isfile(MODEL_PATH_BEST)
    if model_exists:
        checkpoint = torch.load(MODEL_PATH_BEST)
        net.load_state_dict(checkpoint['state_dict'])
        print ("MODEL LOADED, CONTINUING TRAINING")
        return "TRAINING AVG LOSS: {}\n" \
               "TRAINING AVG DIFF: {}".format(
            checkpoint["epoch_avg_loss"], checkpoint["epoch_avg_diff"])
    else:
        if optional:
            pass  # model loading was optional, so nothing to do
        else:
            # shit, no model
            raise Exception("model couldn't be found:", MODEL_PATH_BEST)


loss_function = nn.MSELoss()
if hyperdash_support:
    exp = Experiment("simple lstm - cheetah")
    exp.param("layers", LSTM_LAYERS)
    exp.param("nodes", HIDDEN_NODES)

if TRAIN:
    optimizer = optim.Adam(net.parameters())
    if CONTINUE:
        old_model_string = loadModel(optional=True)
        print(old_model_string)
else:
    old_model_string = loadModel(optional=False)

loss_history = [999999999]  # very high loss because loss can't be empty for min()

for epoch_idx in np.arange(EPOCHS):

    loss_epoch = 0
    diff_epoch = 0

    for episode_idx, data in enumerate(dataloader):
        x, y = makeIntoVariables(data)
        # diff_episode = F.mse_loss(x.data, y.data).data.cpu()[0]

        # reset hidden lstm units
        net.zero_hidden()

        loss_episode = 0
        diff_episode = 0
        if TRAIN:
            optimizer.zero_grad()

        # iterate over episode frames
        for frame_idx in np.arange(len(x)):

            prediction = net.forward(x[frame_idx])
            loss = loss_function(prediction, y[frame_idx].view(1, -1))

            loss_episode += loss.data.cpu()[0]
            diff_episode += F.mse_loss(x[frame_idx].data, y[frame_idx].data).data.cpu()[0]
            if TRAIN:
                loss.backward(retain_graph=True)

        if TRAIN:
            optimizer.step()

        loss.detach_()
        net.hidden[0].detach_()
        net.hidden[1].detach_()

        printEpisodeLoss(epoch_idx, episode_idx, loss_episode, diff_episode, len(x))

        loss_epoch += loss_episode
        diff_epoch += diff_episode

    printEpochLoss(epoch_idx, episode_idx, loss_epoch, diff_epoch)
    if TRAIN:
        saveModel(
            state=net.state_dict(),
            epoch=epoch_idx,
            episode_idx=episode_idx,
            loss_epoch=loss_epoch,
            diff_epoch=diff_epoch,
            is_best=(loss_epoch < min(loss_history))
        )
        loss_history.append(loss_epoch)
    else:
        print(old_model_string)
        break

# Cleanup and mark that the experiment successfully completed
if hyperdash_support:
    exp.end()
