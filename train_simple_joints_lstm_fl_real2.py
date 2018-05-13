import shutil

import numpy as np
from torch import autograd, nn, optim, torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.nn.functional as F

from simple_joints_lstm.dataset_real_smol import DatasetRealSmol
from simple_joints_lstm.lstm_net_real_v3 import LstmNetRealv3
from simple_joints_lstm.lstm_simple_net2_real import LstmSimpleNet2Real
from utils.plot import VisdomExt
import os

try:
    from hyperdash import Experiment

    hyperdash_support = True
except:
    hyperdash_support = False

HIDDEN_NODES = 128
LSTM_LAYERS = 3
EXPERIMENT = 5
EPOCHS = 10
MODEL_PATH = "./trained_models/lstm_real_v2_exp{}_l{}_n{}.pt".format(
    EXPERIMENT,
    LSTM_LAYERS,
    HIDDEN_NODES
)
MODEL_PATH_BEST = "./trained_models/lstm_real_v2_exp{}_l{}_n{}_best.pt".format(
    EXPERIMENT,
    LSTM_LAYERS,
    HIDDEN_NODES
)
# TRAIN = True
# CONTINUE = False
BATCH_SIZE = 1
VIZ = False

dataset_train = DatasetRealSmol(train=True)
dataset_test = DatasetRealSmol(train=False)

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


def extract(dataslice):
    x, y, epi = (Variable(dataslice["x"]),
                 Variable(dataslice["y"]),
                 dataslice["epi"].numpy()[0])

    if torch.cuda.is_available():
        x = x.cuda()
        y = y.cuda()

    return x, y, epi


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
    if is_best:
        shutil.copyfile(MODEL_PATH, MODEL_PATH_BEST)


#
#
# def loadModel(optional=True):
#     model_exists = os.path.isfile(MODEL_PATH_BEST)
#     if model_exists:
#         checkpoint = torch.load(MODEL_PATH_BEST)
#         net.load_state_dict(checkpoint['state_dict'])
#         print("MODEL LOADED, CONTINUING TRAINING")
#         return "TRAINING AVG LOSS: {}\n" \
#                "TRAINING AVG DIFF: {}".format(
#             checkpoint["epoch_avg_loss"], checkpoint["epoch_avg_diff"])
#     else:
#         if optional:
#             pass  # model loading was optional, so nothing to do
#         else:
#             # shit, no model
#             raise Exception("model couldn't be found:", MODEL_PATH_BEST)


loss_function = nn.MSELoss()
if hyperdash_support:
    exp = Experiment("[sim2real] lstm - real v3")
    exp.param("exp", EXPERIMENT)
    exp.param("layers", LSTM_LAYERS)
    exp.param("nodes", HIDDEN_NODES)

# if TRAIN:
optimizer = optim.Adam(net.parameters())
# if CONTINUE:
#     old_model_string = loadModel(optional=True)
#     print(old_model_string)
# else:
#     old_model_string = loadModel(optional=False)

loss_history = [np.inf]  # very high loss because loss can't be empty for min()

for epoch in np.arange(EPOCHS):

    loss_epoch = 0
    diff_epoch = 0

    epi_x_old = 0
    x_buf = []
    y_buf = []

    for epi, data in enumerate(dataloader_train):
        x, y, epi_x = extract(data)

        net.zero_grad()
        net.zero_hidden()
        optimizer.zero_grad()

        if epi_x != epi_x_old or epi == len(dataset_train) - 1:
            x_cat = torch.cat(x_buf, 0).unsqueeze(1)
            y_cat = torch.cat(y_buf, 0).unsqueeze(1)

            delta = net.forward(x_cat)

            loss = loss_function(x_cat[:, :, :12] + delta, y_cat)
            loss.backward()
            optimizer.step()

            x_buf = []
            y_buf = []
            epi_x_old = epi_x

            loss_episode = loss.clone().cpu().data.numpy()[0]
            diff_episode = F.mse_loss(x_cat[:, :, :12], y_cat).clone().cpu().data.numpy()[0]

            loss.detach_()
            net.hidden[0].detach_()
            net.hidden[1].detach_()

            if exp is not None:
                exp.metric("loss episode", loss_episode)
                exp.metric("diff episode", diff_episode)
                exp.metric("epoch", epoch)

            loss_epoch += loss_episode
            diff_epoch += diff_episode

        x_buf.append(x)
        y_buf.append(y)

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
            loss = loss_function(x_cat[:, :, :12] + delta, y_cat)

            loss_total.append(loss.clone().cpu().data.numpy()[0])
            diff_total.append(F.mse_loss(x_cat[:, :, :12], y_cat).clone().cpu().data.numpy()[0])

            x_buf = []
            y_buf = []
            epi_x_old = epi_x

        x_buf.append(x)
        y_buf.append(y)



    if hyperdash_support:
        exp.metric("loss test mean", np.mean(loss_total))
        exp.metric("diff test mean", np.mean(diff_total))
        exp.metric("epoch", epoch)

# Cleanup and mark that the experiment successfully completed
if hyperdash_support:
    exp.end()
