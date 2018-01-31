import shutil

import numpy as np
from torch import autograd, nn, optim, torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from fuel.streams import DataStream
from fuel.schemes import ShuffledScheme
from fuel.datasets.hdf5 import H5PYDataset

# absolute imports here, so that you can run the file directly
from simple_joints_lstm.lstm_simple_net2_pusher import LstmSimpleNet2Pusher
from simple_joints_lstm.mujoco_dataset_pusher3dof import MujocoPusher3DofDataset
from simple_joints_lstm.params_adrien import *
from utils.plot import VisdomExt
import os

try:
    from hyperdash import Experiment

    hyperdash_support = True
except:
    hyperdash_support = False

# batch size has to be 1, otherwise the LSTM doesn't know what to do
# dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=1)
train_data = H5PYDataset(
    DATASET_PATH, which_sets=('train',), sources=('s_transition_obs','r_transition_obs')
)
stream_train = DataStream(train_data, iteration_scheme=ShuffledScheme(train_data.num_examples, 1))

net = LstmSimpleNet2Pusher(8)

print(net)

if CUDA:
    net.cuda()

if TRAIN:
    print("STARTING IN TRAINING MODE")
else:
    print("STARTING IN VALIDATION MODE")

viz = VisdomExt([["loss"],["diff"]],[dict(title='LSTM loss', xlabel='iteration', ylabel='loss'),
dict(title='Diff loss', xlabel='iteration', ylabel='error')])


def makeIntoVariables(dataslice):
    # import ipdb; ipdb.set_trace()
    # if CUDA:
    #     return x.cuda()[0], y.cuda()[0]
    x, y = autograd.Variable(
        # Don't predict palet and goal position
        torch.from_numpy(dataslice["s_transition_obs"][:,:,:-4]).cuda(),
        requires_grad=False
    ), autograd.Variable(
        torch.from_numpy(dataslice["r_transition_obs"][:,:,:-4]).cuda(),
        requires_grad=False
    )

    return x, y  # because we have minibatch_size=1


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
    exp = Experiment("simple lstm - pusher")
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
    iterator = stream_train.get_epoch_iterator(as_dict=True)

    for episode_idx, data in enumerate(iterator):
        x, y = makeIntoVariables(data)

        # reset hidden lstm units
        net.zero_grad()
        net.zero_hidden()
        optimizer.zero_grad()

        correction = net.forward(x)
        loss = loss_function(x+correction, y).mean()
        loss.backward()

        optimizer.step()

        loss_episode = loss.clone().cpu().data.numpy()[0]
        diff_episode = F.mse_loss(x, y).clone().cpu().data.numpy()[0]
        # import ipdb; ipdb.set_trace()
        printEpisodeLoss(epoch_idx, episode_idx, loss_episode, diff_episode, 100)
        viz.update(epoch_idx*train_data.num_examples+episode_idx, loss_episode, "loss")
        viz.update(epoch_idx*train_data.num_examples+episode_idx, diff_episode, "diff")

        loss_epoch += loss_episode
        diff_epoch += diff_episode
        loss.detach_()
        net.hidden[0].detach_()
        net.hidden[1].detach_()

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
