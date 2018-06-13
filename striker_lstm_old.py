import shutil
import sys

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


max_steps = int(sys.argv[1]) or 10000
print("Training lstm with: {} datapoints".format(max_steps))
HIDDEN_NODES = 256
LSTM_LAYERS = 3
dropout = 0.3
EPOCHS = 250
DATASET_PATH_REL = "/data/lisa/data/sim2real/"
DATASET_PATH = DATASET_PATH_REL + "mujoco_striker_trained_{}.h5".format(max_steps)
MODEL_PATH = "./trained_models/lstm_striker_{}l_{}_trained_{}.pt".format(
    LSTM_LAYERS,
    HIDDEN_NODES,
    # dropo\u,
    max_steps
)
MODEL_PATH_BEST = "./trained_models/lstm_striker_{}l_{}_trained_{}_best.pt".format(
    LSTM_LAYERS,
    HIDDEN_NODES,
    # dropout,
    max_steps
)
TRAIN = True
CONTINUE = False
CUDA = True
print(MODEL_PATH_BEST)
batch_size = 8
train_data = H5PYDataset(
    DATASET_PATH, which_sets=('train',), sources=('s_transition_obs','r_transition_obs', 'obs', 'actions')
)
stream_train = DataStream(train_data, iteration_scheme=ShuffledScheme(train_data.num_examples, batch_size))
valid_data = H5PYDataset(
    DATASET_PATH, which_sets=('valid',), sources=('s_transition_obs','r_transition_obs', 'obs', 'actions')
)
stream_valid = DataStream(valid_data, iteration_scheme=SequentialScheme(valid_data.num_examples, batch_size))
data = next(stream_train.get_epoch_iterator(as_dict=True))
net = LstmStriker(53, 14, use_cuda=CUDA, batch=batch_size,
    hidden_nodes=HIDDEN_NODES, lstm_layers=LSTM_LAYERS, dropouti=dropout)
print(net)
if CUDA:
    net.cuda()
viz = VisdomExt([["loss", "validation loss"],["diff"]],[dict(title='LSTM loss', xlabel='iteration', ylabel='loss'),
dict(title='Diff loss', xlabel='iteration', ylabel='error')])

means = {
    'o': np.array([1.12845600e+00, 2.85855811e-02, 6.05528504e-02, -9.97073829e-01, 6.56818366e-03, 1.40352142e+00,3.84055846e-03, 1.02492295e-01,
        7.11518712e-03, 4.65221144e-02, -2.46240869e-01, 1.76974421e-03, 3.31832051e-01, -6.37228310e-04, 2.20609486e-01, -2.38019347e-01,
        8.07869807e-02, 4.95883346e-01, -1.74267560e-01, -2.71298617e-01, 4.24245656e-01, 5.50004303e-01, -3.22469383e-01], dtype='float32'),
    's': np.array([9.59522069e-01, 2.64616702e-02, -1.47129979e-03, -1.00690019e+00, 8.95156711e-03, -2.26142392e-01, 4.37198719e-03, -3.18137455e+00,
        -8.15565884e-02, -1.13330793e+00, -1.15818921e-02, 3.88979465e-02,-3.10147667e+01, 1.01360520e-02, 2.72106588e-01, -1.99255228e-01, 8.47409219e-02, 4.95847106e-01,
        -1.74359173e-01, -2.71298617e-01, 4.24228728e-01, 5.50020635e-01, -3.22452545e-01], dtype='float32'),
    'c': np.array([1.74270540e-01, 2.45837355e-03, 6.43057898e-02, -2.44984333e-03, -2.27138959e-03, 1.64594924e+00, -5.81401051e-04, 3.28941011e+00, 8.86482969e-02, 1.17964220e+00, -2.34902933e-01, -3.72229367e-02, 3.13308201e+01, -1.16425110e-02], dtype='float32'),
}

std = {
    'o': np.array([0.81256843, 0.25599328, 1.04780889, 0.78443158, 0.98964226, 0.66285115, 0.99393243, 1.09946322, 0.49455127, 2.67891765,
         2.24620128, 3.26377749, 2.13994312, 3.29273677, 4.68308300e-01, 2.50030875e-01, 1.48312315e-01, 5.18861376e-02, 5.50917946e-02,
         1.29705272e-03, 1.62378117e-01, 2.58738667e-01, 2.16630325e-02], dtype='float32'),
    's': np.array([0.57472217, 0.24750113, 1.07578814, 0.78113234, 0.99487597, 0.11505181, 1.00299263, 6.00020313,
        1.16489089, 4.44655132, 4.39647961, 3.27095556, 14.71065998, 3.32328629, 3.99973363e-01, 2.70193309e-01, 1.47384644e-01,
        5.26564866e-02, 5.60375415e-02, 1.29705272e-03,1.62536576e-01, 2.58769482e-01, 2.21352559e-02], dtype='float32'),
    'a': np.array([1.73093116, 1.73142111, 1.73274243, 1.73260796, 1.73203647, 1.73140657, 1.7309823], dtype='float32'),
    'c': np.array([3.13339353e-01, 3.22756357e-02, 1.76987484e-01, 2.07477063e-01, 1.05615752e-02, 7.54027247e-01, 1.01595912e-02, 5.85010672e+00, 1.06995499e+00, 3.47300959e+00, 3.72587252e+00, 3.02333444e-01, 1.42488728e+01, 3.63102883e-01], dtype='float32'),
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
    viz.update(epoch, np.mean(diff_epoch), "diff")
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
