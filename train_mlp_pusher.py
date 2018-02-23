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
from utils.plot import VisdomExt
import os


HIDDEN_NODES = 128
LSTM_LAYERS = 3
EXPERIMENT = 1
EPOCHS = 200
DATASET_PATH_REL = "/data/lisa/data/sim2real/"
# DATASET_PATH_REL = "/lindata/sim2real/"
DATASET_PATH = DATASET_PATH_REL + "mujoco_data_pusher3dof_5ac_backl.h5"
MODEL_PATH = "./trained_models/mlp_pusher_5ac_{}l_{}_ep{}.pt".format(
    LSTM_LAYERS,
    HIDDEN_NODES,
    EPOCHS
)
MODEL_PATH_BEST = "./trained_models/mlp_pusher_5ac_{}l_{}_ep{}_best.pt".format(
    LSTM_LAYERS,
    HIDDEN_NODES,
    EPOCHS
)
TRAIN = True
CONTINUE = False
CUDA = True

batch_size = 1
train_data = H5PYDataset(
    DATASET_PATH, which_sets=('train',), sources=('s_transition_obs','r_transition_obs', 'obs', 'actions')
)
stream_train = DataStream(train_data, iteration_scheme=ShuffledScheme(train_data.num_examples, batch_size))
valid_data = H5PYDataset(
    DATASET_PATH, which_sets=('valid',), sources=('s_transition_obs','r_transition_obs')
)
stream_valid = DataStream(train_data, iteration_scheme=ShuffledScheme(train_data.num_examples, batch_size))

net = torch.nn.Sequential(
    torch.nn.Linear(15, HIDDEN_NODES),
    nn.LeakyReLU(0.2),
    torch.nn.Linear(HIDDEN_NODES, HIDDEN_NODES),
    nn.LeakyReLU(0.2),
    torch.nn.Linear(HIDDEN_NODES, HIDDEN_NODES/2),
    nn.LeakyReLU(0.2),
    torch.nn.Linear(HIDDEN_NODES/2, 6),
)

print(net)

if CUDA:
    net.cuda()

viz = VisdomExt([["loss", "validation loss"],["diff"]],[dict(title='LSTM loss', xlabel='iteration', ylabel='loss'),
dict(title='Diff loss', xlabel='iteration', ylabel='error')])


def makeIntoVariables(dat):
    input_ = np.concatenate([dat["obs"][:,:,:6], dat["actions"], dat["s_transition_obs"][:,:,:6]], axis=2)
    x, y = autograd.Variable(
        # Don't predict palet and goal position
        torch.from_numpy(input_).cuda(),
        requires_grad=False
    ), autograd.Variable(
        torch.from_numpy(dat["r_transition_obs"][:,:,:6]).cuda(),
        requires_grad=False
    )
    return x, y



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

loss_min = float('inf')  # very high loss because loss can't be empty for min()

for epoch in np.arange(EPOCHS):

    loss_epoch = 0
    diff_epoch = 0
    iterator = stream_train.get_epoch_iterator(as_dict=True)

    for epi, data in enumerate(iterator):
        x, y = makeIntoVariables(data)

        # reset hidden lstm units
        net.zero_grad()
        optimizer.zero_grad()

        correction = net.forward(x)
        loss = loss_function(x[:,:,-6:]+correction, y).mean()
        loss.backward()

        optimizer.step()

        loss_episode = loss.clone().cpu().data.numpy()[0]
        diff_episode = F.mse_loss(x[:,:,-6:], y).clone().cpu().data.numpy()[0]
        # printEpisodeLoss(epoch, epi, loss_episode, diff_episode, 100)
        viz.update(epoch*train_data.num_examples+epi, loss_episode, "loss")
        viz.update(epoch*train_data.num_examples+epi, diff_episode, "diff")

        loss_epoch += loss_episode
        diff_epoch += diff_episode
        loss.detach_()
        net.hidden[0].detach_()
        net.hidden[1].detach_()

    printEpochLoss(epoch, epi, loss_epoch, diff_epoch)


    # Validation step
    loss_total = []
    iterator = stream_valid.get_epoch_iterator(as_dict=True)
    for epi, data in enumerate(iterator):
        x, y = makeIntoVariables(data)
        net.zero_hidden()
        correction = net.forward(x)
        loss = loss_function(x[:,:,-6:]+correction, y).mean()
        loss_total.append(loss.clone().cpu().data.numpy()[0])
    loss_valid = np.mean(loss_total)
    viz.update(epoch*train_data.num_examples, loss_valid, "validation loss")

    if TRAIN:
        saveModel(
            state=net.state_dict(),
            epoch=epoch,
            episode_idx=epi,
            loss_epoch=loss_epoch,
            diff_epoch=diff_epoch,
            is_best=(loss_valid <= loss_min)
        )
        loss_min = min(loss_min, loss_valid)
    else:
        print(old_model_string)
        break

# Cleanup and mark that the experiment successfully completed
if hyperdash_support:
    exp.end()
