import shutil

import numpy as np
from torch import nn, optim, torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.nn.functional as F
from fuel.streams import DataStream
from fuel.schemes import ShuffledScheme, SequentialScheme
from fuel.datasets.hdf5 import H5PYDataset

# absolute imports here, so that you can run the file directly
from simple_joints_lstm.pusher_lstm import LstmSimpleNet2Pusher
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
EPOCHS = 350
DATASET_PATH_REL = "/data/lisa/data/sim2real/"
# DATASET_PATH_REL = "/lindata/sim2real/"
DATASET_PATH = DATASET_PATH_REL + "mujoco_data_pusher3dof_big_backl.h5"
MODEL_PATH = "./trained_models/lstm_pusher_{}l_{}_overfit.pt".format(
    LSTM_LAYERS,
    HIDDEN_NODES
)
MODEL_PATH_BEST = "./trained_models/lstm_pusher_{}l_{}_best.pt".format(
    LSTM_LAYERS,
    HIDDEN_NODES
)
TRAIN = True
CONTINUE = False
CUDA = True
print(MODEL_PATH_BEST)
batch_size = 1
train_data = H5PYDataset(
    DATASET_PATH, which_sets=('train',), sources=('s_transition_obs','r_transition_obs', 'obs', 'actions')
)
stream_train = DataStream(train_data, iteration_scheme=ShuffledScheme(train_data.num_examples, batch_size))
valid_data = H5PYDataset(
    DATASET_PATH, which_sets=('valid',), sources=('s_transition_obs','r_transition_obs', 'obs', 'actions')
)
stream_valid = DataStream(valid_data, iteration_scheme=SequentialScheme(valid_data.num_examples, batch_size))

net = LstmSimpleNet2Pusher(27, 6)
print(net)

if CUDA:
    net.cuda()

viz = VisdomExt([["loss", "validation loss"],["diff"]],[dict(title='LSTM loss', xlabel='iteration', ylabel='loss'),
dict(title='Diff loss', xlabel='iteration', ylabel='error')])

means = {
    'o': np.array([2.2281456, 1.93128324, 1.63007331, 0.48472479, 0.4500702, 0.30325469, -0.38825685, -0.63075501, 0.63863981, -0.63173348, 1.01628101, -1.02707994], dtype='float32'),
    's': np.array([2.25090551, 1.94997263, 1.6495719, 0.43379614, 0.3314755, 0.43763939, -0.38825685, -0.63075501, 0.63863981, -0.63173348, 1.01628101, -1.02707994], dtype='float32'),
    'c': np.array([0.00173789, 0.00352129, -0.00427585, 0.05105286, 0.11881274, -0.13443381], dtype='float32')
    # 'r': np.array([2.25277853,  1.95338345, 1.64534044, 0.48487723, 0.45031613, 0.30320421], dtype='float32')
}

std = {
    'o': np.array([0.56555426, 0.5502255 , 0.59792095, 1.30218685, 1.36075258, 2.37941241, 0.37797412, 0.50458646, 0.24756368, 0.1611862, 0.11527784, 0.11197065], dtype='float32'),
    's': np.array([0.5295766, 0.51998389, 0.57609886, 1.35480666, 1.40806067, 2.43865967, 0.36436939, 0.50140637, 0.24752532, 0.16115381, 0.11527784, 0.11197065], dtype='float32'),
    'c': np. array([0.01608515, 0.0170644, 0.01075647, 0.46635619, 0.53578401, 0.32062387], dtype='float32')
    # 'r':  np.array([0.52004296, 0.51547343, 0.57784373, 1.30222356, 1.36113203, 2.38046765], dtype='float32')
}

def makeIntoVariables(dat):
    input_ = np.concatenate([
        (dat["obs"] - means['o']) / std['o'],
        dat["actions"],
        (dat["s_transition_obs"] - means['s']) / std['s']
    ], axis=2)
    x, y = Variable(
        torch.from_numpy(input_).cuda(),
        requires_grad=False
    ), Variable(
        torch.from_numpy(dat["r_transition_obs"][:,:,:6]).cuda(),
        requires_grad=False
    )
    return x, y

def printEpochLoss(epoch_idx, valid, loss_epoch, diff_epoch):
    print("epoch {}, "
          "loss: {}, , "
          "diff: {}, valid loss: {}".format(
        epoch_idx,
        round(loss_epoch, 4),
        round(diff_epoch, 4),
        round(valid, 4)
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
    optimizer = optim.Adam(net.parameters(), lr=0.0001)
    if CONTINUE:
        old_model_string = loadModel(optional=True)
        print(old_model_string)
else:
    old_model_string = loadModel(optional=False)

loss_min = [float('inf')]
iterator = stream_train.get_epoch_iterator(as_dict=True)
data = next(iterator)
import ipdb; ipdb.set_trace()
for epoch in np.arange(EPOCHS):
    loss_epoch = []
    diff_epoch = []
    iterator = stream_train.get_epoch_iterator(as_dict=True)

    for epi, _ in enumerate(iterator):
        x, y = makeIntoVariables(data)

        # reset hidden lstm units
        net.zero_grad()
        net.zero_hidden()
        optimizer.zero_grad()

        correction = net.forward(x)
        sim_prediction = Variable(torch.from_numpy(data["s_transition_obs"][:,:,:6]), requires_grad=False).cuda()
        loss = loss_function(sim_prediction+correction, y).mean()
        loss.backward()

        optimizer.step()

        loss_episode = loss.clone().cpu().data.numpy()[0]
        diff_episode = F.mse_loss(sim_prediction, y).clone().cpu().data.numpy()[0]

        loss_epoch.append(loss_episode)
        diff_epoch.append(diff_episode)
        loss.detach_()
        net.hidden[0].detach_()
        net.hidden[1].detach_()

    viz.update(epoch, np.mean(loss_epoch), "loss")
    viz.update(epoch, np.mean(diff_episode), "diff")


    # Validation step
    # loss_valid = []
    # iterator = stream_valid.get_epoch_iterator(as_dict=True)
    # for _, data in enumerate(iterator):
    #     x, y = makeIntoVariables(data)
    #     net.zero_hidden()
    #     correction = net.forward(x)
    #     sim_prediction = Variable(torch.from_numpy(data["s_transition_obs"][:,:,:6]), requires_grad=False).cuda()
    #     loss = loss_function(sim_prediction+correction, y).mean()
    #     loss_valid.append(loss.clone().cpu().data.numpy()[0])
    # loss_valid = np.mean(loss_valid)
    # viz.update(epoch*train_data.num_examples, loss_valid, "validation loss")
    #
    # printEpochLoss(epoch, loss_valid, loss_epoch, diff_epoch)
    printEpochLoss(epoch, 0, loss_epoch, diff_epoch)
    #
    # if TRAIN:
    saveModel(
        state=net.state_dict(),
        epoch=epoch,
        episode_idx=epi,
        loss_epoch=loss_epoch,
        diff_epoch=diff_epoch,
        is_best=False
        # is_best=(loss_valid < loss_min)
    )
    #     loss_min = min(loss_valid, loss_min)
    # else:
    #     print(old_model_string)
    #     break

# Cleanup and mark that the experiment successfully completed
if hyperdash_support:
    exp.end()
