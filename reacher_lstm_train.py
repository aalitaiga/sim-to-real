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
from simple_joints_lstm.reacher_lstm import LstmReacher
# from simple_joints_lstm.params_adrien import *
from utils.plot import VisdomExt
import os


HIDDEN_NODES = 128
LSTM_LAYERS = 3
EXPERIMENT = 1
EPOCHS = 150
DATASET_PATH_REL = "/data/lisa/data/sim2real/"
# DATASET_PATH_REL = "/lindata/sim2real/"
DATASET_PATH = DATASET_PATH_REL + "mujoco_reacher.h5"
# DATASET_PATH = "/Tmp/alitaiga/mujoco_reacher_test.h5"
MODEL_PATH = "./trained_models/reacher/lstm_reacher_{}l_{}.pt".format(
    LSTM_LAYERS,
    HIDDEN_NODES
)
MODEL_PATH_BEST = "./trained_models/reacher/lstm_reacher_{}l_{}_best.pt".format(
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
    DATASET_PATH, which_sets=('valid',), sources=SequentialScheme('s_transition_obs','r_transition_obs', 'obs', 'actions')
)
stream_valid = DataStream(valid_data, iteration_scheme=(valid_data.num_examples, batch_size))
data = next(stream_train.get_epoch_iterator(as_dict=True))
net = LstmReacher(22, 4, normalized=False, use_cuda=CUDA)
print(net)
# import ipdb; ipdb.set_trace()
if CUDA:
    net.cuda()

viz = VisdomExt([["loss", "validation loss"],["diff"]],[dict(title='LSTM loss', xlabel='iteration', ylabel='loss'),
dict(title='Diff loss', xlabel='iteration', ylabel='error')])

means = {
    'o': np.array([7.51461267e-01, 1.41610339e-01, 4.60824937e-01, 7.13317981e-03, 1.03738275e-03, -1.48758627e-04, 1.62978816e+00, -1.20971948e-02, 9.29422453e-02,3.90605591e-02], dtype='float32'),
    's': np.array([7.30040550e-01, 1.17301859e-01, 4.79943186e-01, 6.87805656e-03, 1.03738275e-03, -1.48758627e-04, 1.59260523e+00, -1.71832982e-02, 8.88551399e-02, 4.05287929e-02], dtype='float32'),
    'c': np.array([8.47268850e-02,  -1.57003014e-05], dtype='float32'),
    'theta': np.array([8.49875738e-04, -1.58988897e-07], dtype='float32'),
}

std = {
    'o': np.array([0.30590999, 0.78053445, 0.35970449, 0.60881543, 0.11474513, 0.11735817, 0.8761031, 6.51498461, 0.1432249 , 0.13590805], dtype='float32'),
    's': np.array([0.32542008, 0.77895236, 0.36168751, 0.61597532, 0.11474513, 0.11735817, 2.44059658, 6.59672832, 0.14335664, 0.1362942], dtype='float32'),
    'a': np.array([0.57733983, 0.57892293], dtype='float32'),
    'c': np.array([2.28456879e+00, 5.76027029e-04], dtype='float32'),
    'theta': np.array([2.28884127e-02, 5.80418919e-06], dtype='float32')
}

def makeIntoVariables(dat):
    input_ = np.concatenate([
        (dat["obs"][:,:,:10] - means['o']) / std['o'],
        (dat["actions"] / std['a']),
        (dat["s_transition_obs"][:,:,:10] - means['s']) / std['s']
    ], axis=2)
    x, y, theta_r = Variable(
        torch.from_numpy(input_).cuda(),
        requires_grad=False
    ), Variable(
        torch.from_numpy(
            dat["r_transition_obs"][:,:,6:8]
        ).cuda(),
        requires_grad=False
    ), Variable(
        torch.from_numpy(
            np.arccos(dat["r_transition_obs"][:,:,:2]) * np.sign(dat["r_transition_obs"][:,:,2:4])
        ).cuda(),
        requires_grad=False
    )
    return x, y, theta_r

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


loss_function = nn.MSELoss()

if TRAIN:
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    scheduler = MultiStepLR(optimizer, milestones=[50,75,100], gamma=0.5)
    if CONTINUE:
        old_model_string = loadModel(optional=True)
        print(old_model_string)
else:
    old_model_string = loadModel(optional=False)

loss_min = [float('inf')]
m_c = torch.from_numpy(means["c"])
s_c = torch.from_numpy(std["c"])
m_t = torch.from_numpy(means["theta"])
s_t = torch.from_numpy(std["theta"])
# if CUDA:
#     m_c.cuda(), s_c.cuda(), m_t.cuda(), s_t.cuda()
mean_c, std_c = Variable(m_c, requires_grad=False).cuda(), Variable(s_c, requires_grad=False).cuda()
mean_t, std_t = Variable(m_t, requires_grad=False).cuda(), Variable(s_t, requires_grad=False).cuda()

for epoch in np.arange(EPOCHS):
    loss_epoch = []
    diff_epoch = []
    iterator = stream_train.get_epoch_iterator(as_dict=True)

    for epi, data in enumerate(iterator):
        # assert np.isclose(data["obs"][:,:,0], np.sign(data["obs"][:,:,4]) * np.arccos(data["obs"][:,:,2]), atol=35e-5).all()
        # assert np.isclose(np.arccos(data["obs"][:,:,:2]), np.sign(data["obs"][:,:,4]) * np.arccos(data["obs"][:,:,2])).all()
        # np.isclose(np.arccos(data["obs"][:,:,:2]) * np.sign(data["obs"][:,:,2])).all()
        x, y, theta_r = makeIntoVariables(data)

        # reset hidden lstm units
        net.zero_grad()
        net.zero_hidden()
        optimizer.zero_grad()

        correction = net.forward(x)
        velocities = Variable(torch.from_numpy(
            data["s_transition_obs"][:,:,6:8]
        ), requires_grad=False).cuda()
        theta_s = Variable(torch.from_numpy(
            np.arccos(data["s_transition_obs"][:,:,:2]) * np.sign(data["s_transition_obs"][:,:,2:4])
        ), requires_grad=False).cuda()
        # import ipdb; ipdb.set_trace()
        loss = loss_function(correction[:,:,2:4], (y-velocities - mean_c) / std_c).mean() + \
            loss_function(correction[:,:,:2], (theta_r - theta_s - mean_t) / std_t).mean()
        loss.backward()
        optimizer.step()

        loss_episode = loss.clone().cpu().data.numpy()[0]
        diff_episode = (F.mse_loss(correction[:,:,2:4], y) + F.mse_loss(correction[:,:,:2], theta_r)).clone().cpu().data.numpy()[0]

        loss_epoch.append(loss_episode)
        diff_epoch.append(diff_episode)
        loss.detach_()
        net.hidden[0].detach_()
        net.hidden[1].detach_()

    viz.update(epoch, np.mean(loss_epoch), "loss")
    viz.update(epoch, np.mean(diff_epoch), "diff")
    scheduler.step()


    # Validation step
    loss_valid = []
    iterator = stream_valid.get_epoch_iterator(as_dict=True)
    for _, data in enumerate(iterator):
        x, y, theta_r = makeIntoVariables(data)
        net.zero_hidden()

        correction = net.forward(x)
        velocities = Variable(torch.from_numpy(
            data["s_transition_obs"][:,:,6:8]
        ), requires_grad=False).cuda()
        theta_s = Variable(torch.from_numpy(
            np.arccos(data["s_transition_obs"][:,:,:2]) * np.sign(data["s_transition_obs"][:,:,2:4])
        ), requires_grad=False).cuda()
        loss = loss_function(correction[:,:,2:4], (y-velocities - mean_c) / std_c).mean() + \
            loss_function(correction[:,:,:2], (theta_r - theta_s - mean_t) / std_t).mean()
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
