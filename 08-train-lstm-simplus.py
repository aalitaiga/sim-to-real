import torch
import numpy as np
import os
import matplotlib.pyplot as plt
import gym
import gym_ergojr
from hyperdash import Experiment
from torch.autograd import Variable
from tqdm import tqdm
import torch.nn.functional as F
from simple_joints_lstm.lstm_net_real_v3 import LstmNetRealv3

DEBUG = False
# DEBUG = True
HIDDEN_NODES = 128
LSTM_LAYERS = 3
EXPERIMENT = 1
EPOCHS = 5
MODEL_PATH = "./trained_models/lstm_real_vX3_exp{}_l{}_n{}.pt".format(
    EXPERIMENT,
    LSTM_LAYERS,
    HIDDEN_NODES
)

ds = np.load(os.path.expanduser("~/data/sim2real/data-realigned-v2-{}-bullet.npz".format("train")))

ds_curr_real = ds["ds_curr_real"]
ds_next_real = ds["ds_next_real"]
ds_next_sim = ds["ds_next_sim"]
ds_action = ds["ds_action"]
ds_epi = ds["ds_epi"]

env = gym.make("ErgoFightStatic-Headless-Shield-Move-HalfRand-Bullet-Plus-Training-v0")
env.set_net(nodes=HIDDEN_NODES, layers=LSTM_LAYERS)
_ = env.reset()
old_obs = ds_curr_real[0]
env.set_state(ds_curr_real[0])


def saveModel(states):
    print("saving...")
    torch.save({"state_dict": states}, MODEL_PATH)
    print("saved to",MODEL_PATH)


optimizer = torch.optim.Adam(env.get_parameters())
loss_function = torch.nn.MSELoss()


exp = Experiment("[sim2real] lstm-real v5-directSim+")
exp.param("exp", EXPERIMENT)
exp.param("layers", LSTM_LAYERS)
exp.param("nodes", HIDDEN_NODES)

def tmp_var(data):
    return Variable(torch.from_numpy(data), volatile=True)

for epoch in range(EPOCHS):

    losses = Variable(torch.zeros(1)).cuda()
    variables_to_delete = []
    epi_x_old = 0

    loss_epi = 0
    diff_epi = 0

    for epi in tqdm(range(len(ds_curr_real))):

        if ds_epi[epi] != epi_x_old or epi == len(ds_curr_real) - 1:
            losses.backward()
            optimizer.step()

            losses.detach_()
            # cleanup
            for element in variables_to_delete:
                del element
            del losses
            variables_to_delete = []
            losses = Variable(torch.zeros(1)).cuda()

            env.net.hidden[0].detach_()
            env.net.hidden[1].detach_()
            env.net.zero_grad()
            env.net.zero_hidden()
            optimizer.zero_grad()

            epi_x_old = ds_epi[epi]

            exp.metric("loss episode", loss_epi)
            exp.metric("diff episode", diff_epi)
            exp.metric("epoch", epoch)

            loss_epi = 0
            diff_epi = 0


        else:  # only calculate loss when not in the middle of a transition

            action = ds_action[epi]
            _, _, _, info = env.step(action)
            new_obs = info["new_state"]

            if DEBUG:
                print("real t1_x:", np.around(ds_curr_real[epi], 2))
                print("sim_ t1_x:", np.around(old_obs[:12], 2))
                print("real t2_x:", np.around(ds_next_real[epi], 2))
                print("sim_ t2_x:", np.around(new_obs.data.cpu().numpy()[0, 0], 2))
                print("===")

            target = Variable(torch.from_numpy(ds_next_real[epi])).cuda()

            loss = loss_function(new_obs, target)
            losses += loss
            loss_epi += loss.clone().cpu().data.numpy()[0]
            diff_epi += F.mse_loss(tmp_var(ds_curr_real[epi]), tmp_var(ds_next_real[epi])).clone().cpu().data.numpy()[0]

        env.set_state(ds_next_real[epi])
        old_obs = ds_next_real[epi]

    saveModel(env.net.state_dict())
