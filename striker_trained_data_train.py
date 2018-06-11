import shutil
import sys

import argparse
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
from simple_joints_lstm.striker_lstm import LstmStriker
# from simple_joints_lstm.params_adrien import *
from utils.plot import VisdomExt
import os

parser = argparse.ArgumentParser(description='LSTM training for Striker')
parser.add_argument('wdrop', default=0, type=bool)
parser.add_argument('dropout', default='0.0', type=float)
args = parser.parse_args()

max_steps = 10000 #int(sys.argv[1]) or
print("Training lstm with: {} datapoints".format(max_steps))
HIDDEN_NODES = 256
LSTM_LAYERS = 3
EPOCHS = 200
DATASET_PATH_REL = "/data/lisa/data/sim2real/striker/"
DATASET_PATH = DATASET_PATH_REL + "mujoco_striker_trained_{}_2.h5".format(max_steps)
MODEL_PATH = "./trained_models/striker_l{}_h{}_d{}_wdrop{}_trained_{}.pt".format(
    LSTM_LAYERS,
    HIDDEN_NODES,
    args.dropout,
    args.wdrop,
    max_steps,
)
MODEL_PATH_BEST = "./trained_models/striker_l{}_h{}_d{}_wdrop{}_trained_{}_best.pt".format(
    LSTM_LAYERS,
    HIDDEN_NODES,
    args.dropout,
    args.wdrop,
    max_steps,
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
    DATASET_PATH, which_sets=('valid',), sources=SequentialScheme('s_transition_obs','r_transition_obs', 'obs', 'actions')
)
stream_valid = DataStream(valid_data, iteration_scheme=(valid_data.num_examples, batch_size))
data = next(stream_train.get_epoch_iterator(as_dict=True))
net = LstmStriker(53, 14, use_cuda=CUDA, batch=batch_size,
    hidden_nodes=HIDDEN_NODES, lstm_layers=LSTM_LAYERS, wdrop=False, dropouti=args.dropout)
print(net)
# import ipdb; ipdb.set_trace()
if CUDA:
    net.cuda()
viz = VisdomExt([["loss", "validation loss"],["diff"]],[dict(title='LSTM loss', xlabel='iteration', ylabel='loss'),
dict(title='Diff loss', xlabel='iteration', ylabel='error')])

# save normalization
if not os.path.isfile('normalization/striker_trained/mean_obs.npy'):
    assert data["obs"].shape[0] == train_data.num_examples
    np.save('normalization/striker_trained/mean_obs.npy', data["obs"].mean(axis=(0, 1)))
    np.save('normalization/striker_trained/mean_s_transition_obs.npy', data["s_transition_obs"].mean(axis=(0,1)))
    np.save('normalization/striker_trained/mean_correction.npy', (data["r_transition_obs"] - data["s_transition_obs"])[:,:,:14].mean(axis=(0,1)))
    np.save('normalization/striker_trained/std_obs.npy', data["obs"].std(axis=(0,1)))
    np.save('normalization/striker_trained/std_actions.npy', data["actions"].std(axis=(0,1)))
    np.save('normalization/striker_trained/std_s_transition_obs.npy', data["s_transition_obs"].std(axis=(0,1)))
    np.save('normalization/striker_trained/std_correction.npy', (data["r_transition_obs"] - data["s_transition_obs"])[:,:,:14].std(axis=(0,1)))
    os._exit(0)
else:
    means = {
        'o': np.load('normalization/striker_trained/mean_obs.npy'),
        's': np.load('normalization/striker_trained/mean_s_transition_obs.npy'),
        'c': np.load('normalization/striker_trained/mean_correction.npy'),
    }

    std = {
        'o': np.load('normalization/striker_trained/std_obs.npy'),
        'a': np.load('normalization/striker_trained/std_actions.npy'),
        's': np.load('normalization/striker_trained/std_s_transition_obs.npy'),
        'c': np.load('normalization/striker_trained/std_correction.npy'),
    }

def makeIntoVariables(dat):
    input_ = np.concatenate([
        (dat["obs"][:,:,:] - means['o']) / std['o'],
        (dat["actions"] / std['a']),
        (dat["s_transition_obs"][:,:,:] - means['s']) / std['s']
    ], axis=2)
    x, y = Variable(
        torch.from_numpy(input_).cuda(),
        requires_grad=False
    ), Variable(
        torch.from_numpy(
            dat["r_transition_obs"][:,:,:14]
        ).cuda(),
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


def saveModel(state, epoch, loss_epoch, valid_epoch, is_best, episode_idx):
    torch.save({
        "epoch": epoch,
        "episodes": episode_idx + 1,
        "state_dict": state,
        "epoch_avg_loss": round(loss_epoch, 10),
        "epoch_avg_valid": round(valid_epoch, 10)
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


loss_function = nn.MSELoss().cuda()

if TRAIN:
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    scheduler = MultiStepLR(optimizer, milestones=[60,85,110], gamma=0.5)
    if CONTINUE:
        old_model_string = loadModel(optional=True)
        print(old_model_string)
else:
    old_model_string = loadModel(optional=False)

loss_min = [float('inf')]
m_c = torch.from_numpy(means["c"])
s_c = torch.from_numpy(std["c"])
mean_c, std_c = Variable(m_c, requires_grad=False).cuda(), Variable(s_c, requires_grad=False).cuda()

for epoch in np.arange(EPOCHS):
    loss_epoch = []
    diff_epoch = []
    iterator = stream_train.get_epoch_iterator(as_dict=True)
    net.train()

    for epi, data in enumerate(iterator):
        x, y = makeIntoVariables(data)

        # reset hidden lstm units
        net.zero_grad()
        net.zero_hidden()
        optimizer.zero_grad()

        correction = net(x)
        sim_prediction = Variable(torch.from_numpy(data["s_transition_obs"][:,:,:14]), requires_grad=False).cuda()
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
    net.eval()
    loss_valid = []
    iterator = stream_valid.get_epoch_iterator(as_dict=True)
    for _, data in enumerate(iterator):
        x, y = makeIntoVariables(data)
        net.zero_hidden()
        correction = net(x)
        sim_prediction = Variable(torch.from_numpy(data["s_transition_obs"][:,:,:14]), requires_grad=False).cuda()
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
            valid_epoch=loss_val,
            is_best=(loss_val < loss_min)
        )
        loss_min = min(loss_val, loss_min)
    else:
        print(old_model_string)
        break
