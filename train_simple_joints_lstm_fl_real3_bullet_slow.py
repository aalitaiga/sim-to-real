import shutil

import numpy as np
from torch import nn, optim, torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.nn.functional as F

from simple_joints_lstm.dataset_real_smol_bullet import DatasetRealSmolBullet
from simple_joints_lstm.lstm_net_real_v3 import LstmNetRealv3

try:
    from hyperdash import Experiment

    hyperdash_support = True
except:
    hyperdash_support = False

HIDDEN_NODES = 128
LSTM_LAYERS = 3
EXPERIMENT = 5
EPOCHS = 5
MODEL_PATH = "./trained_models/lstm_real_v6_exp{}_l{}_n{}.pt".format(
    EXPERIMENT,
    LSTM_LAYERS,
    HIDDEN_NODES
)
MODEL_PATH_BEST = "./trained_models/lstm_real_v6_exp{}_l{}_n{}_best.pt".format(
    EXPERIMENT,
    LSTM_LAYERS,
    HIDDEN_NODES
)
# TRAIN = True
# CONTINUE = False
BATCH_SIZE = 1
VIZ = False

dataset_train = DatasetRealSmolBullet(train=True)
dataset_test = DatasetRealSmolBullet(train=False)

# batch size has to be 1, otherwise the LSTM doesn't know what to do
dataloader_train = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=False, num_workers=1)
dataloader_test = DataLoader(dataset_test, batch_size=BATCH_SIZE, shuffle=False, num_workers=1)

net = LstmNetRealv3(
    n_input_state_sim=12,
    n_input_state_real=12,
    n_input_actions=6,
    nodes=HIDDEN_NODES,
    layers=LSTM_LAYERS)

if torch.cuda.is_available():
    net = net.cuda()

net = net.float()


def extract(dataslice):
    x, y, epi = (Variable(dataslice["x"]).float(),
                 Variable(dataslice["y"]).float(),
                 dataslice["epi"].numpy()[0])

    if torch.cuda.is_available():
        x = x.cuda()
        y = y.cuda()

    return x.unsqueeze(0), y.unsqueeze(0), epi


def printEpochLoss(epoch_idx, epoch_len, loss_epoch, diff_epoch):
    loss_avg = round(float(loss_epoch) / (epoch_len+np.finfo(np.float32).eps), 2)
    diff_avg = round(float(diff_epoch) / (epoch_len+np.finfo(np.float32).eps), 2)
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
    if is_best:
        shutil.copyfile(MODEL_PATH, MODEL_PATH_BEST)


loss_function = nn.MSELoss()
if hyperdash_support:
    exp = Experiment("[sim2real] lstm - real v3")
    exp.param("exp", EXPERIMENT)
    exp.param("layers", LSTM_LAYERS)
    exp.param("nodes", HIDDEN_NODES)

optimizer = optim.Adam(net.parameters())

loss_history = [np.inf]  # very high loss because loss can't be empty for min()

for epoch in np.arange(EPOCHS):

    loss_epoch = 0
    diff_epoch = 0

    epi_x_old = 0
    x_buf = []
    y_buf = []

    loss_buffer = Variable(torch.zeros(1))

    for epi, data in enumerate(dataloader_train):
        x, y, epi_x = extract(data)

        delta = net.forward(x)
        loss = loss_function(delta, y)
        loss_buffer += loss

        loss_episode = loss.clone().cpu().data.numpy()[0]
        diff_episode = F.mse_loss(x[:, :, :12], x[:, :, :12] + y).clone().cpu().data.numpy()[0]
        loss_epoch += loss_episode
        diff_epoch += diff_episode

        if epi_x != epi_x_old or epi == len(dataset_train) - 1:
            loss.backward()
            optimizer.step()

            loss.detach_()
            net.hidden[0].detach_()
            net.hidden[1].detach_()
            net.zero_grad()
            net.zero_hidden()
            optimizer.zero_grad()
            epi_x_old = epi_x

            if exp is not None:
                exp.metric("loss episode", loss_episode)
                exp.metric("diff episode", diff_episode)
                exp.metric("epoch", epoch)


    print("loss_epoch", loss_epoch)
    printEpochLoss(epoch, epi_x_old, loss_epoch, diff_epoch)
    saveModel(
        state=net.state_dict(),
        epoch=epoch,
        epoch_len=epi_x_old,
        loss_epoch=loss_epoch,
        diff_epoch=diff_epoch,
        is_best=(loss_epoch < min(loss_history))
    )
    loss_history.append(loss_epoch)

    # Validation step
    loss_total = []
    diff_total = []

    epi_x_old = 0
    x_buf = []
    y_buf = []
    for epi, data in enumerate(dataloader_test):
        x, y, epi_x = extract(data)
        net.zero_hidden()

        if epi_x != epi_x_old or epi == len(dataset_test) - 1:
            x_cat = torch.cat(x_buf, 0).unsqueeze(1)
            y_cat = torch.cat(y_buf, 0).unsqueeze(1)

            delta = net.forward(x_cat)
            loss = loss_function(delta, y_cat)

            loss_total.append(loss.clone().cpu().data.numpy()[0])
            diff_total.append(F.mse_loss(x_cat[:, :, :12], x_cat[:, :, :12] + y_cat).clone().cpu().data.numpy()[0])

            x_buf = []
            y_buf = []
            epi_x_old = epi_x

        x_buf.append(x.squeeze(0))
        y_buf.append(y.squeeze(0))

    if hyperdash_support:
        exp.metric("loss test mean", np.mean(loss_total))
        exp.metric("diff test mean", np.mean(diff_total))
        exp.metric("epoch", epoch)

# Cleanup and mark that the experiment successfully completed
if hyperdash_support:
    exp.end()
