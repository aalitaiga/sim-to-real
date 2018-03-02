import shutil

import numpy as np
from torch import autograd, nn, optim, torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from fuel.streams import DataStream
from fuel.schemes import ShuffledScheme
from fuel.datasets.hdf5 import H5PYDataset

# absolute imports here, so that you can run the file directly
from simple_joints_lstm.dataset_real import DatasetRealPosVel
from simple_joints_lstm.lstm_simple_net2_real import LstmSimpleNet2Real
# from simple_joints_lstm.params_adrien import *
from utils.plot import VisdomExt
import os

try:
    from hyperdash import Experiment

    hyperdash_support = True
except:
    hyperdash_support = False

HIDDEN_NODES = 128
LSTM_LAYERS = 3
EXPERIMENT = 1
EPOCHS = 10
DATASET_PATH = "/windata/sim2real-full/done/"
MODEL_PATH = "./trained_models/simple_lstm_real_norm{}_v1_{}l_{}.pt".format(
    EXPERIMENT,
    LSTM_LAYERS,
    HIDDEN_NODES
)
MODEL_PATH_BEST = "./trained_models/simple_lstm_real{}_v1_{}l_{}_best.pt".format(
    EXPERIMENT,
    LSTM_LAYERS,
    HIDDEN_NODES
)
TRAIN = True
CONTINUE = False
CUDA = True
BATCH_SIZE = 1
VIZ = False

dataset_train = DatasetRealPosVel(DATASET_PATH, for_training=True, with_velocities=True, normalized=True)
dataset_test = DatasetRealPosVel(DATASET_PATH, for_training=False, with_velocities=True, normalized=True)

# batch size has to be 1, otherwise the LSTM doesn't know what to do
dataloader_train = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=1)
dataloader_test = DataLoader(dataset_test, batch_size=BATCH_SIZE, shuffle=True, num_workers=1)

net = LstmSimpleNet2Real(n_input_state_sim=12, n_input_state_real=12, n_input_actions=6)

if CUDA:
    net.cuda()

if VIZ:
    viz = VisdomExt([["loss", "validation loss"], ["diff"]],
                    [dict(title='LSTM loss', xlabel='iteration', ylabel='loss'),
                     dict(title='Diff loss', xlabel='iteration', ylabel='error')])


def makeIntoVariables(dataslice):
    x, y = (autograd.Variable(
        dataslice["state_next_sim_joints"][:, :, :].cuda(),
        requires_grad=False
    ), autograd.Variable(
        dataslice["state_current_real_joints"][:, :, :].cuda(),
        requires_grad=False
    ), autograd.Variable(
        dataslice["action"][:, :, :].cuda(),
        requires_grad=False
    )), autograd.Variable(
        dataslice["state_next_real_joints"][:, :, :].cuda(),
        requires_grad=False
    )
    return x, y


def printEpisodeLoss(epoch_idx, episode_idx, loss_episode, diff_episode, len_episode):
    loss_avg = round(float(loss_episode) / len_episode, 2)
    diff_avg = round(float(diff_episode) / len_episode, 2)
    # print("epoch {}, episode {}, "
    #       "loss: {}, loss avg: {}, "
    #       "diff: {}, diff avg: {}".format(
    #     epoch_idx,
    #     episode_idx,
    #     round(loss_episode, 2),
    #     loss_avg,
    #     round(diff_episode, 2),
    #     diff_avg
    # ))
    if hyperdash_support:
        exp.metric("diff avg", diff_avg)
        exp.metric("loss avg", loss_avg)


def printEpochLoss(epoch_idx, episode_idx, loss_epoch, diff_epoch):
    loss_avg = round(float(loss_epoch) / (episode_idx + 1), 2)
    diff_avg = round(float(diff_epoch) / (episode_idx + 1), 2)
    print("epoch {}, "
          "loss: {}, loss avg: {}, "
          "diff: {}, diff avg: {}".format(
        epoch_idx,
        round(loss_epoch, 2),
        loss_avg,
        round(diff_epoch, 2),
        diff_avg
    ))

    if hyperdash_support:
        exp.metric("epoch", epoch_idx)
        exp.metric("diff train epoch avg", diff_avg)
        exp.metric("loss train epoch avg", loss_avg)


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
        print("MODEL LOADED, CONTINUING TRAINING")
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
    exp = Experiment("simple lstm - real")
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

for epoch in np.arange(EPOCHS):

    loss_epoch = 0
    diff_epoch = 0

    for epi, data in enumerate(dataloader_train):
        x, y = makeIntoVariables(data)

        # reset hidden lstm units
        net.zero_grad()
        net.zero_hidden()
        optimizer.zero_grad()

        correction = net.forward(x)
        loss = loss_function(x[0] + correction, y).mean()
        loss.backward()

        optimizer.step()

        loss_episode = loss.clone().cpu().data.numpy()[0]
        diff_episode = F.mse_loss(x[0], y).clone().cpu().data.numpy()[0]
        # printEpisodeLoss(epoch, epi, loss_episode, diff_episode, 100)
        if VIZ:
            viz.update(epoch * 1290 + epi, loss_episode, "loss")
            viz.update(epoch * 1290 + epi, diff_episode, "diff")
        if hyperdash_support:
            exp.metric("loss train", loss_episode)
            exp.metric("diff train", diff_episode)
            exp.metric("epoch", epoch)

        loss_epoch += loss_episode
        diff_epoch += diff_episode
        loss.detach_()
        net.hidden[0].detach_()
        net.hidden[1].detach_()

    printEpochLoss(epoch, epi, loss_epoch, diff_epoch)
    if TRAIN:
        saveModel(
            state=net.state_dict(),
            epoch=epoch,
            episode_idx=epi,
            loss_epoch=loss_epoch,
            diff_epoch=diff_epoch,
            is_best=(loss_epoch < min(loss_history))
        )
        loss_history.append(loss_epoch)
    else:
        print(old_model_string)
        break

    # Validation step
    loss_total = []
    diff_total = []

    for epi, data in enumerate(dataloader_test):
        x, y = makeIntoVariables(data)
        net.zero_hidden()
        correction = net.forward(x)
        loss = loss_function(x[0] + correction, y).mean()
        loss_total.append(loss.clone().cpu().data.numpy()[0])
        diff_total.append(F.mse_loss(x[0], y).clone().cpu().data.numpy()[0])
    if VIZ:
        viz.update(epoch * 1290, np.mean(loss_total), "validation loss")
    if hyperdash_support:
        exp.metric("loss test mean", np.mean(loss_total))
        exp.metric("diff test mean", np.mean(diff_total))
        exp.metric("epoch", epoch)

# Cleanup and mark that the experiment successfully completed
if hyperdash_support:
    exp.end()
