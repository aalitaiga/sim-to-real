import shutil

import numpy as np
from torch import nn, optim, torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm

from simple_joints_lstm.dataset_real_smol_bullet_v2 import DatasetRealSmolBulletV2
from simple_joints_lstm.dataset_real_smol_bullet_v2_nosim import DatasetRealSmolBulletV2NoSim
from simple_joints_lstm.lstm_net_real_v3 import LstmNetRealv3

try:
    from hyperdash import Experiment

    hyperdash_support = True
except:
    hyperdash_support = False

HIDDEN_NODES = 128
LSTM_LAYERS = 3
EXPERIMENT = 1
EPOCHS = 5
MODEL_PATH = "./trained_models/lstm_real_nosim_vX4_exp{}_l{}_n{}.pt".format(
    EXPERIMENT,
    LSTM_LAYERS,
    HIDDEN_NODES
)
# TRAIN = True
# CONTINUE = False
BATCH_SIZE = 1
VIZ = False

dataset_train = DatasetRealSmolBulletV2NoSim(train=True)
dataset_test = DatasetRealSmolBulletV2NoSim(train=False)

# batch size has to be 1, otherwise the LSTM doesn't know what to do
dataloader_train = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=False, num_workers=1)
dataloader_test = DataLoader(dataset_test, batch_size=BATCH_SIZE, shuffle=False, num_workers=1)

net = LstmNetRealv3(
    n_input_state_sim=0,
    n_input_state_real=12,
    n_input_actions=6,
    nodes=HIDDEN_NODES,
    layers=LSTM_LAYERS)

if torch.cuda.is_available():
    net = net.cuda()

net = net.float()


def extract(dataslice):
    x, y = (Variable(dataslice["x"].transpose(0,1)).float(),
                 Variable(dataslice["y"].transpose(0,1)).float())

    if torch.cuda.is_available():
        x = x.cuda()
        y = y.cuda()

    return x, y


def printEpochLoss(epoch_idx, epoch_len, loss_epoch, diff_epoch):
    loss_avg = round(float(loss_epoch) / epoch_len, 2)
    diff_avg = round(float(diff_epoch) / epoch_len, 2)
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
        exp.metric("loss train epoch avg", loss_avg)
        exp.metric("diff train epoch avg", diff_avg)


def saveModel(state, epoch, loss_epoch, diff_epoch, is_best, epoch_len):
    torch.save({
        "epoch": epoch,
        "epoch_len": epoch_len,
        "state_dict": state,
        "epoch_avg_loss": float(loss_epoch) / epoch_len,
        "epoch_avg_diff": float(diff_epoch) / epoch_len
    }, MODEL_PATH)


loss_function = nn.MSELoss()
if hyperdash_support:
    exp = Experiment("[sim2real] lstm-realv4")
    exp.param("exp", EXPERIMENT)
    exp.param("layers", LSTM_LAYERS)
    exp.param("nodes", HIDDEN_NODES)

optimizer = optim.Adam(net.parameters())

loss_history = [np.inf]  # very high loss because loss can't be empty for min()

for epoch in np.arange(EPOCHS):

    loss_epoch = 0
    diff_epoch = 0

    for epi_idx, epi_data in enumerate(dataloader_train):
        x, y = extract(epi_data)

        net.zero_grad()
        net.zero_hidden()
        optimizer.zero_grad()

        delta = net.forward(x)

        # for idx in range(len(x)):
        #     print(idx, "=")
        #     print("real t1_x:", np.around(x[idx, 0, 12:24].cpu().data.numpy(), 2))
        #     print("sim_ t2_x:", np.around(x[idx, 0, :12].cpu().data.numpy(), 2))
        #     print("action__x:", np.around(x[idx, 0, 24:].cpu().data.numpy(), 2))
        #     print("real t2_x:",
        #           np.around(x[idx, 0, :12].cpu().data.numpy() + y[idx, 0].cpu().data.numpy(), 2))
        #     print("real t2_y:",
        #           np.around(x[idx, 0, :12].cpu().data.numpy() + delta[idx, 0].cpu().data.numpy(), 2))
        #     print("delta___x:",
        #           np.around(y[idx, 0].cpu().data.numpy(), 3))
        #     print("delta___y:",
        #           np.around(delta[idx, 0].cpu().data.numpy(), 3))
        #     print("===")

        loss = loss_function(delta, y)
        loss.backward()
        optimizer.step()

        loss_episode = loss.clone().cpu().data.numpy()[0]
        diff_episode = F.mse_loss(x[:, :, :12], x[:, :, :12] + y).clone().cpu().data.numpy()[0]

        loss.detach_()
        net.hidden[0].detach_()
        net.hidden[1].detach_()

        if exp is not None:
            exp.metric("loss episode", loss_episode)
            exp.metric("diff episode", diff_episode)
            exp.metric("epoch", epoch)

        loss_epoch += loss_episode
        diff_epoch += diff_episode

    printEpochLoss(epoch, epi_idx, loss_epoch, diff_epoch)
    saveModel(
        state=net.state_dict(),
        epoch=epoch,
        epoch_len=epi_idx,
        loss_epoch=loss_epoch,
        diff_epoch=diff_epoch,
        is_best=(loss_epoch < min(loss_history))
    )
    loss_history.append(loss_epoch)

    # Validation step
    loss_total = []
    diff_total = []

    for epi_idx, epi_data in enumerate(dataloader_test):
        x, y = extract(epi_data)
        net.zero_hidden()

        delta = net.forward(x)
        loss = loss_function(delta, y)

        loss_total.append(loss.clone().cpu().data.numpy()[0])
        diff_total.append(F.mse_loss(x[:, :, :12], x[:, :, :12] + y).clone().cpu().data.numpy()[0])


    if hyperdash_support:
        exp.metric("loss test mean", np.mean(loss_total))
        exp.metric("diff test mean", np.mean(diff_total))
        exp.metric("epoch", epoch)

# Cleanup and mark that the experiment successfully completed
if hyperdash_support:
    exp.end()
