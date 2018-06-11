import shutil

import numpy as np
from torch import nn, optim, torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm

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
EPOCHS = 2
MODEL_PATH = "./trained_models/lstm_real_vT_exp{}_l{}_n{}.pt".format(
    EXPERIMENT,
    LSTM_LAYERS,
    HIDDEN_NODES
)
MODEL_PATH_BEST = "./trained_models/lstm_real_vT_exp{}_l{}_n{}_best.pt".format(
    EXPERIMENT,
    LSTM_LAYERS,
    HIDDEN_NODES
)
# TRAIN = True
# CONTINUE = False
BATCH_SIZE = 1
VIZ = False

dataset_test = DatasetRealSmolBullet(train=False)

# batch size has to be 1, otherwise the LSTM doesn't know what to do
dataloader_test = DataLoader(dataset_test, batch_size=BATCH_SIZE, shuffle=False, num_workers=1)

net_concat = LstmNetRealv3(
    n_input_state_sim=12,
    n_input_state_real=12,
    n_input_actions=6,
    nodes=HIDDEN_NODES,
    layers=LSTM_LAYERS)
net_lotsabackprop = LstmNetRealv3(
    n_input_state_sim=12,
    n_input_state_real=12,
    n_input_actions=6,
    nodes=HIDDEN_NODES,
    layers=LSTM_LAYERS)
net_lossadd = LstmNetRealv3(
    n_input_state_sim=12,
    n_input_state_real=12,
    n_input_actions=6,
    nodes=HIDDEN_NODES,
    layers=LSTM_LAYERS)

if torch.cuda.is_available():
    net_concat = net_concat.cuda()
    net_lotsabackprop = net_lotsabackprop.cuda()
    net_lossadd = net_lossadd.cuda()

net_concat = net_concat.float()
net_lotsabackprop = net_lotsabackprop.float()
net_lossadd = net_lossadd.float()


def extract(dataslice):
    x, y, epi = (Variable(dataslice["x"]).float(),
                 Variable(dataslice["y"]).float(),
                 dataslice["epi"].numpy()[0])

    if torch.cuda.is_available():
        x = x.cuda()
        y = y.cuda()

    return x.unsqueeze(0), y.unsqueeze(0), epi


def printEpochLoss(epoch_idx, epoch_len, loss_epoch, diff_epoch):
    loss_avg = round(float(loss_epoch) / (epoch_len + np.finfo(np.float32).eps), 2)
    diff_avg = round(float(diff_epoch) / (epoch_len + np.finfo(np.float32).eps), 2)
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


loss_function = nn.MSELoss()
if hyperdash_support:
    exp = Experiment("[sim2real] lstm - real v3 - test")
    exp.param("exp", EXPERIMENT)
    exp.param("layers", LSTM_LAYERS)
    exp.param("nodes", HIDDEN_NODES)

optimizer_concat = optim.Adam(net_concat.parameters())
optimizer_lotsabackprop = optim.Adam(net_lotsabackprop.parameters())
optimizer_lossadd = optim.Adam(net_lossadd.parameters())

loss_history = [np.inf]  # very high loss because loss can't be empty for min()

for epoch in np.arange(EPOCHS):

    loss_epoch_la = 0
    loss_epoch_lb = 0
    loss_epoch_cc = 0
    diff_epoch = 0

    epi_x_old = 0
    x_buf = []
    y_buf = []

    loss_buffer = Variable(torch.zeros(1))
    if torch.cuda.is_available():
        loss_buffer = loss_buffer.cuda()

    for epi, data in tqdm(enumerate(dataloader_test)):
        x, y, epi_x = extract(data)

        delta = net_lossadd.forward(x)
        loss_lossadd = loss_function(delta, y)
        loss_episode_la = loss_lossadd.clone().cpu().data.numpy()[0]
        loss_buffer += loss_lossadd

        delta = net_lotsabackprop.forward(x)
        loss_lotsabackprop = loss_function(delta, y)
        loss_episode_lb = loss_lotsabackprop.clone().cpu().data.numpy()[0]
        loss_lotsabackprop.backward(retain_graph=True)
        optimizer_lotsabackprop.step()

        diff_episode = F.mse_loss(x[:, :, :12], x[:, :, :12] + y).clone().cpu().data.numpy()[0]
        loss_epoch_la += loss_episode_la
        loss_epoch_lb += loss_episode_lb

        diff_epoch += diff_episode

        if epi_x != epi_x_old or epi == len(dataset_test) - 1:
            loss_lossadd.backward()
            optimizer_lossadd.step()

            x_cat = torch.cat(x_buf, 0).unsqueeze(1)
            y_cat = torch.cat(y_buf, 0).unsqueeze(1)

            delta = net_concat.forward(x_cat)
            loss_concat = loss_function(delta, y_cat)
            loss_episode_cc = loss_concat.clone().cpu().data.numpy()[0]
            loss_concat.backward()
            optimizer_concat.step()
            loss_epoch_cc += loss_episode_cc

            for loss, net, optimizer in [
                (loss_lossadd, net_lossadd, optimizer_lossadd),
                (loss_concat, net_concat, optimizer_concat),
                (loss_lotsabackprop,net_lotsabackprop,optimizer_lotsabackprop)]:
                loss.detach_()
                net.hidden[0].detach_()
                net.hidden[1].detach_()
                net.zero_grad()
                net.zero_hidden()
                optimizer.zero_grad()
                loss_buffer.zero_()

            del loss_concat

            x_buf = []
            y_buf = []
            epi_x_old = epi_x

            if exp is not None:
                exp.metric("loss episode la", loss_episode_la)
                exp.metric("loss episode lb", loss_episode_lb)
                exp.metric("loss episode cc", loss_episode_cc)
                exp.metric("diff episode", diff_episode)
                exp.metric("epoch", epoch)

        x_buf.append(x.squeeze(0))
        y_buf.append(y.squeeze(0))


    # Validation step
    loss_total_la = []
    loss_total_lb = []
    loss_total_cc = []
    diff_total = []

    epi_x_old = 0
    x_buf = []
    y_buf = []
    for epi, data in enumerate(dataloader_test):
        x, y, epi_x = extract(data)

        delta_lossadd = net_lossadd.forward(x)
        loss_lossadd = loss_function(delta_lossadd, y)

        delta_lotsabackpropo = net_lotsabackprop.forward(x)
        loss_lotsabackprop = loss_function(delta_lotsabackpropo, y)

        delta_concat = net_concat.forward(x)
        loss_concat = loss_function(delta_concat, y)

        loss_total_la.append(loss_lossadd.clone().cpu().data.numpy()[0])
        loss_total_lb.append(loss_lotsabackprop.clone().cpu().data.numpy()[0])
        loss_total_cc.append(loss_concat.clone().cpu().data.numpy()[0])
        diff_total.append(F.mse_loss(x[:, :, :12], x[:, :, :12] + y).clone().cpu().data.numpy()[0])

        if epi_x != epi_x_old or epi == len(dataset_test) - 1:


            epi_x_old = epi_x
            for loss, net in [
                (loss_lossadd, net_lossadd),
                (loss_concat, net_concat),
                (loss_lotsabackprop,net_lotsabackprop)]:
                net.hidden[0].detach_()
                net.hidden[1].detach_()
                net.zero_grad()
                net.zero_hidden()
                del loss_lotsabackprop
                del loss_concat
                del loss_lossadd


    if hyperdash_support:
        exp.metric("loss test mean LA", np.mean(loss_total_la))
        exp.metric("loss test mean LB", np.mean(loss_total_lb))
        exp.metric("loss test mean CC", np.mean(loss_total_cc))
        exp.metric("diff test mean", np.mean(diff_total))
        exp.metric("epoch", epoch)

# Cleanup and mark that the experiment successfully completed
if hyperdash_support:
    exp.end()






#  75
# | loss episode la:   0.510200 |
# | loss episode lb:   2.033563 |
# | loss episode cc:   0.041948 |
# | diff episode:   0.493939 |
# | epoch:   0.000000 |