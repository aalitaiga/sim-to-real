import shutil

import numpy as np
from torch import nn, optim, torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR
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
EPOCHS = 250
DATASET_PATH_REL = "/data/lisa/data/sim2real/"
# DATASET_PATH_REL = "/lindata/sim2real/"
DATASET_PATH = DATASET_PATH_REL + "mujoco_pusher3dof_small.h5"
MODEL_PATH = "./trained_models/real_only/lstm_pusher_{}l_{}.pt".format(
    LSTM_LAYERS,
    HIDDEN_NODES
)
MODEL_PATH_BEST = "./trained_models/real_only/lstm_pusher_{}l_{}_best.pt".format(
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
stream_train = DataStream(train_data, iteration_scheme=SequentialScheme(train_data.num_examples, batch_size))
valid_data = H5PYDataset(
    DATASET_PATH, which_sets=('valid',), sources=('s_transition_obs','r_transition_obs', 'obs', 'actions')
)
stream_valid = DataStream(valid_data, iteration_scheme=ShuffledScheme(valid_data.num_examples, batch_size))

net = LstmSimpleNet2Pusher(15, 10)
print(net)
# import ipdb; ipdb.set_trace()
if CUDA:
    net.cuda()

viz = VisdomExt([["loss", "validation loss"],["diff"]],[dict(title='LSTM loss', xlabel='iteration', ylabel='loss'),
dict(title='Diff loss', xlabel='iteration', ylabel='error')])

means = {
    'o': np.array([-0.4417094, 1.50765455, -0.02639891, -0.05560728, 0.39159551, 0.03819341, 0.76052153, 0.23057458, 0.63315856, -0.6400153, 1.01691067, -1.02684915], dtype='float32'),
    's': np.array([-0.44221497, 1.52240622, -0.02244471, 0.01573334, 0.23615479, 0.10089023, 0.7594685, 0.23817146, 0.63317519, -0.64011943, 1.01691067, -1.02684915], dtype='float32'),
    # 'c': np.array([-2.23746197e-03, 4.93022148e-03, -2.03814497e-03, -6.97841570e-02, 1.53955221e-01, -6.21460043e-02], dtype='float32')
    'c': np.array([-2.73653027e-03, 1.95451882e-02, 1.91621704e-03, 1.56128232e-03, -1.51736499e-03, 5.49889286e-04, -1.03732746e-03, 7.78057473e-03, 1.73114568e-05, -9.45877109e-05], dtype='float32')
    # 'r': np.array([2.25277853,  1.95338345, 1.64534044, 0.48487723, 0.45031613, 0.30320421], dtype='float32')
}

std = {
    'o': np.array([0.38327965, 0.78956741, 0.48310387, 0.33454728, 0.53120506, 0.51319438, 0.20692779, 0.36664706, 0.25205335, 0.15865214, 0.11554158, 0.1132608], dtype='float32'),
    's': np.array([0.38500383, 0.78036022, 0.48781601, 0.35502997, 0.60374367, 0.56180185, 0.21046612, 0.36828887, 0.25209084, 0.15857539, 0.11554158, 0.1132608], dtype='float32'),
    # 'c': np.array([7.19802594e-03, 1.59114692e-02, 7.24539673e-03, 2.23035514e-01, 4.93483037e-01, 2.18238667e-01,], dtype='float32'),
    'c': np.array([0.01655727, 0.02646242, 0.02456561, 0.11201099, 0.1206677, 0.31924954, 0.00993428, 0.00796531, 0.00071493, 0.00133473], dtype='float32'),
    'a': np.array([0.57690412, 0.57732242, 0.57705152], dtype='float32')
    # 'r':  np.array([0.52004296, 0.51547343, 0.57784373, 1.30222356, 1.36113203, 2.38046765], dtype='float32')
}

def makeIntoVariables(dat):
    input_ = np.concatenate([
        (dat["obs"] - means['o']) / std['o'],
        (dat["actions"] / std['a']),
        # (dat["s_transition_obs"] - means['s']) / std['s']
    ], axis=2)
    x, y = Variable(
        torch.from_numpy(input_).cuda(),
        requires_grad=False
    ), Variable(
        torch.from_numpy(dat["r_transition_obs"][:,:,:10]).cuda(),
        requires_grad=False
    )
    return x, y

def printEpochLoss(epoch_idx, valid, loss_epoch, diff_epoch):
    print("epoch {}, "
          "loss: {}, , "
          "diff: {}, valid loss: {}".format(
        epoch_idx,
        round(loss_epoch, 10),
        round(diff_epoch, 10),
        round(valid, 10)
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
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    scheduler = MultiStepLR(optimizer, milestones=[100,175,200], gamma=0.5)
    if CONTINUE:
        old_model_string = loadModel(optional=True)
        print(old_model_string)
else:
    old_model_string = loadModel(optional=False)

loss_min = [float('inf')]
mean_c = Variable(torch.from_numpy(means["c"]), requires_grad=False).cuda()
std_c = Variable(torch.from_numpy(std["c"]), requires_grad=False).cuda()

for epoch in np.arange(EPOCHS):
    loss_epoch = []
    diff_epoch = []
    iterator = stream_train.get_epoch_iterator(as_dict=True)

    for epi, data in enumerate(iterator):
        x, y = makeIntoVariables(data)

        # reset hidden lstm units
        net.zero_grad()
        net.zero_hidden()
        optimizer.zero_grad()

        correction = net.forward(x)
        sim_prediction = Variable(torch.from_numpy(data["obs"][:,:,:10]), requires_grad=False).cuda()
        loss = loss_function(correction, (y-sim_prediction - mean_c) / std_c).mean()
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
    scheduler.step()


    # Validation step
    loss_valid = []
    iterator = stream_valid.get_epoch_iterator(as_dict=True)
    for _, data in enumerate(iterator):
        x, y = makeIntoVariables(data)
        net.zero_hidden()
        correction = net.forward(x)
        sim_prediction = Variable(torch.from_numpy(data["obs"][:,:,:10]), requires_grad=False).cuda()
        loss = loss_function(correction, (y - sim_prediction - mean_c) / std_c).mean()
        loss_valid.append(loss.clone().cpu().data.numpy()[0])
    loss_val = np.mean(loss_valid)
    viz.update(epoch, loss_val, "validation loss")

    printEpochLoss(epoch, loss_val, np.mean(loss_epoch), np.mean(diff_episode))

    if TRAIN:
        saveModel(
            state=net.state_dict(),
            epoch=epoch,
            episode_idx=epi,
            loss_epoch=np.mean(loss_epoch),
            diff_epoch=np.mean(diff_episode),
            is_best=(loss_val < loss_min)
        )
        loss_min = min(loss_val, loss_min)
    else:
        print(old_model_string)
        break

# Cleanup and mark that the experiment successfully completed
if hyperdash_support:
    exp.end()
