import torch
import numpy as np
import gym
import gym_ergojr
from hyperdash import Experiment
from s2rr.movements.dataset import DatasetProduction
from torch.autograd import Variable
from tqdm import tqdm
import torch.nn.functional as F

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

ds = DatasetProduction()
ds.load("~/data/sim2real/data-realigned-v2-{}-bullet.npz".format("train"))

env = gym.make("ErgoFightStatic-Headless-Shield-Move-HalfRand-Bullet-Plus-Training-v0")
env.set_net(nodes=HIDDEN_NODES, layers=LSTM_LAYERS)
_ = env.reset()

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

    for epi in tqdm(range(len(ds.current_real))):

        old_obs = ds.current_real[epi, 0]
        env.set_state(ds.current_real[epi, 0])

        losses = Variable(torch.zeros(1)).cuda()
        loss_epi = 0
        diff_epi = 0

        for frame in range(len(ds.current_real[0])):
            action = ds.action[epi,frame]
            _, _, _, info = env.step(action)
            new_obs = info["new_state"]

            if DEBUG:
                print("real t1_x:", np.around(ds.current_real[epi,frame], 2))
                print("sim_ t1_x:", np.around(old_obs[:12], 2))
                print("real t2_x:", np.around(ds.next_real[epi,frame], 2))
                print("sim_ t2_x:", np.around(new_obs.data.cpu().numpy()[0, 0], 2))
                print("===")

            target = Variable(torch.from_numpy(ds.next_real[epi,frame])).cuda()

            loss = loss_function(new_obs, target)
            losses += loss
            loss_epi += loss.clone().cpu().data.numpy()[0]
            diff_epi += F.mse_loss(tmp_var(ds.current_real[epi,frame]), tmp_var(ds.next_real[epi,frame])).clone().cpu().data.numpy()[0]

        losses.backward()
        optimizer.step()

        losses.detach_()

        del losses
        del loss

        env.net.hidden[0].detach_()
        env.net.hidden[1].detach_()
        env.net.zero_grad()
        env.net.zero_hidden()
        optimizer.zero_grad()

        exp.metric("loss episode", loss_epi)
        exp.metric("diff episode", diff_epi)
        exp.metric("epoch", epoch)

    saveModel(env.net.state_dict())
