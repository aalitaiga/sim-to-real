import time
import numpy as np
import os
import matplotlib.pyplot as plt
import gym
import gym_ergojr
import torch
from gym_ergojr.sim.single_robot import SingleRobot
from hyperdash import Experiment
from s2rr.movements.dataset import DatasetProduction
from torch.autograd import Variable
from simple_joints_lstm.lstm_net_real_v3 import LstmNetRealv3
import torch.nn.functional as F

ds = DatasetProduction()
ds.load("~/data/sim2real/data-realigned-v3-{}-bullet.npz".format("train"))

net = LstmNetRealv3(nodes=128, layers=3)
if torch.cuda.is_available():
    net = net.cuda()

HIDDEN_NODES = 128
LSTM_LAYERS = 3
EXPERIMENT = 1
EPOCHS = 5
MODEL_PATH = "./trained_models/lstm_real_vX5_exp{}_l{}_n{}.pt".format(
    EXPERIMENT,
    LSTM_LAYERS,
    HIDDEN_NODES
)


def double_unsqueeze(data):
    return torch.unsqueeze(torch.unsqueeze(data, dim=0), dim=0)


def double_squeeze(data):
    return torch.squeeze(torch.squeeze(data))


def data_to_var(sim_t2, real_t1, action):
    x = Variable(
        double_unsqueeze(torch.cat(
            [torch.from_numpy(sim_t2).float(),
             torch.from_numpy(real_t1).float(),
             torch.from_numpy(action).float()], dim=0)))
    if torch.cuda.is_available():
        x = x.cuda()
    return x


def to_var(x, volatile=False):
    x = Variable(torch.from_numpy(x), volatile=volatile)
    if torch.cuda.is_available():
        x = x.cuda()
    return x


def save_model(state):
    torch.save({"state_dict": state}, MODEL_PATH)


loss_function = torch.nn.MSELoss()
exp = Experiment("[sim2real] lstm-realv6")
# exp.param("exp", EXPERIMENT)
# exp.param("layers", LSTM_LAYERS)
# exp.param("nodes", HIDDEN_NODES)
optimizer = torch.optim.Adam(net.parameters())

robot = SingleRobot(debug=False)

for epoch in range(EPOCHS):
    for epi in range(len(ds.current_real)):

        net.zero_hidden()  # !important
        net.hidden[0].detach_()  # !important
        net.hidden[1].detach_()  # !important
        net.zero_grad()

        robot.set(ds.current_real[epi, 0])
        robot.act2(ds.current_real[epi, 0, :6])
        robot.step()

        losses = Variable(torch.zeros(1))
        diffs = 0

        for frame in range(len(ds.current_real[0])):
            current_real = robot.observe()
            robot.act2(ds.action[epi, frame])
            robot.step()
            next_sim = robot.observe()

            variable = data_to_var(next_sim, current_real, ds.action[epi, frame])
            delta = double_squeeze(net.forward(variable))
            next_real = to_var(next_sim).float() + delta
            robot.set(next_real.data.cpu().numpy())

            target = to_var(ds.next_real[epi, frame], volatile=True)

            loss = loss_function(next_real, target)
            losses += loss

            diffs += F.mse_loss(to_var(next_sim).float(), to_var(ds.next_real[epi, frame])).clone().cpu().data.numpy()[
                0]

        exp.metric("loss episode", losses.cpu().data.numpy()[0])
        exp.metric("diff episode", diffs)
        exp.metric("epoch", epoch)

        losses.backward()
        optimizer.step()

        del losses
        del loss

    save_model(net.state_dict())

robot.close()
