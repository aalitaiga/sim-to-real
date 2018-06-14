import time

import numpy as np
import os
import matplotlib.pyplot as plt
import gym
import gym_ergojr
import torch
from gym_ergojr.sim.single_robot import SingleRobot
from s2rr.movements.dataset import DatasetProduction
from itertools import cycle

from torch.autograd import Variable

from simple_joints_lstm.lstm_net_real_v3 import LstmNetRealv3
from simple_joints_lstm.lstm_net_real_v4 import LstmNetRealv4

cycol = cycle('bgrcmk')

ds = DatasetProduction()
ds.load("~/data/sim2real/data-realigned-v3-{}-bullet.npz".format("train"))

epi = np.random.randint(0, len(ds.current_real))

joints_sim = np.zeros((299, 6), np.float32)
joints_real = np.zeros((299, 6), np.float32)
joints_simplus = np.zeros((299, 6), np.float32)
joints_nosim = np.zeros((299, 6), np.float32)

modelFile = "../trained_models/lstm_real_vX4_exp1_l3_n128.pt"
net = LstmNetRealv3(nodes=128, layers=3)
full_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), modelFile)
checkpoint = torch.load(full_path, map_location="cpu")
net.load_state_dict(checkpoint['state_dict'])
net.eval()

modelFile = "../trained_models/lstm_real_nosim_vX4_exp1_l3_n128.pt"
net2 = LstmNetRealv3(nodes=128, layers=3,n_input_state_sim=0)
full_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), modelFile)
checkpoint = torch.load(full_path, map_location="cpu")
net2.load_state_dict(checkpoint['state_dict'])
net2.eval()


def double_unsqueeze(data):
    return torch.unsqueeze(torch.unsqueeze(data, dim=0), dim=0)


def double_squeeze(data):
    return torch.squeeze(torch.squeeze(data)).data.cpu().numpy()


def data_to_var(sim_t2, real_t1, action):
    return Variable(
        double_unsqueeze(torch.cat(
            [torch.from_numpy(sim_t2).float(),
             torch.from_numpy(real_t1).float(),
             torch.from_numpy(action).float()], dim=0)), volatile=True)

def data_to_var_nosim(real_t1, action):
    return Variable(
        double_unsqueeze(torch.cat(
            [torch.from_numpy(real_t1).float(),
             torch.from_numpy(action).float()], dim=0)), volatile=True)

#### SIM

robot = SingleRobot(debug=False)
robot.set(ds.current_real[epi, 0])
robot.act2(ds.current_real[epi, 0, :6])
robot.step()
for frame in range(299):
    robot.act2(ds.action[epi, frame])
    robot.step()
    joints_sim[frame, :] = robot.observe()[:6]
    # time.sleep(0.1)
robot.close()

### REAL_SIM

for frame in range(299):
    joints_real[frame, :] = ds.current_real[epi, frame, :6]

#### SIM+

robot = SingleRobot(debug=False)
robot.set(ds.current_real[epi, 0])
robot.act2(ds.current_real[epi, 0, :6])
robot.step()
for frame in range(299):
    old_state = robot.observe()
    robot.act2(ds.action[epi, frame])
    robot.step()
    obs = robot.observe()
    variable = data_to_var(obs, old_state, ds.action[epi, frame])
    delta = double_squeeze(net.forward(variable))
    new_state = obs + 0.7*delta
    robot.set(new_state)

    joints_simplus[frame, :] = new_state[:6]
    # time.sleep(0.1)
robot.close()

#### SIM+2

old_state = ds.current_real[epi, 0]
for frame in range(299):
    variable = data_to_var_nosim(old_state, ds.action[epi, frame])
    new_state = old_state + double_squeeze(net2.forward(variable))
    joints_nosim[frame, :] = new_state[:6]
    old_state = new_state



for i in range(6):
    c = next(cycol)
    plt.plot(
        np.arange(0, 299),
        joints_real[:, i],
        c="red",
        label="real"

    )
    plt.plot(
        np.arange(0, 299),
        joints_sim[:, i],
        c="green",
        dashes=[2, 1],
        label="sim"
    )
    plt.plot(
        np.arange(0, 299),
        ds.action[epi, :, i],
        c="blue",
        dashes=[1, 1],
        label="action"
    )
    plt.plot(
        np.arange(0, 299),
        joints_nosim[:, i],
        c="magenta",
        dashes=[5, 1],
        label="nosim"
    )
    plt.plot(
        np.arange(0, 299),
        joints_simplus[:, i],
        c="black",
        dashes=[1, 4],
        label="simplus"
    )
    plt.legend()
    plt.ylim(-1.25, 1.25)

    plt.show()
