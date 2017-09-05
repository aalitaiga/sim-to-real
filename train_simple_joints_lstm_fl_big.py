import shutil

import numpy as np
from torch import autograd, nn, optim, torch
from torch.utils.data import DataLoader

# absolute imports here, so that you can run the file directly
from simple_joints_lstm.lstm_simple_net import LstmSimpleNet
from simple_joints_lstm.mujoco_traintest_dataset import MujocoTraintestDataset
from simple_joints_lstm.params_adrien import *

dataset = MujocoTraintestDataset(DATASET_PATH, for_training=TRAIN)

batch_size = 16
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1)

net = LstmSimpleNet(batch_size)

if CUDA:
    net.cuda()

if TRAIN:
    print("STARTING IN TRAINING MODE")
else:
    print("STARTING IN VALIDATION MODE")


def makeIntoVariables(dataslice):
    x, y = autograd.Variable(
        dataslice["state_next_sim_joints"],
        requires_grad=False
    ), autograd.Variable(
        dataslice["state_next_real_joints"],
        requires_grad=False
    )
    if CUDA:
        return x.cuda(), y.cuda()
    return x, y

def printEpisodeLoss(epoch_idx, episode_idx, loss_episode, diff_episode, len_episode):
    print("epoch {}, episode {}, "
          "loss: {}, loss avg: {}, "
          "diff: {}, diff avg: {}".format(
        epoch_idx,
        episode_idx,
        round(loss_episode, 4),
        round(float(loss_episode) / len_episode, 2),
        round(diff_episode, 2),
        round(float(diff_episode) / len_episode, 2)
    ))


def printEpochLoss(epoch_idx, episode_idx, loss_epoch, diff_epoch):
    print("epoch {}, "
          "loss: {}, loss avg: {}, "
          "diff: {}, diff avg: {}".format(
        epoch_idx,
        round(loss_epoch, 2),
        round(float(loss_epoch) / (episode_idx + 1), 2),
        round(diff_epoch, 2),
        round(float(diff_epoch) / (episode_idx + 1), 2)
    ))


def saveModel(state, epoch, loss_epoch, diff_epoch, is_best, episode_idx):
    torch.save({
        "epoch": epoch,
        "episodes": episode_idx+1,
        "state_dict": state,
        "epoch_avg_loss": float(loss_epoch) / (episode_idx + 1),
        "epoch_avg_diff": float(diff_epoch) / (episode_idx + 1)
    }, MODEL_PATH)
    if is_best:
        shutil.copyfile(MODEL_PATH, MODEL_PATH_BEST)


def loadModel():
    checkpoint = torch.load(MODEL_PATH_BEST)
    net.load_state_dict(checkpoint['state_dict'])
    return "TRAINING AVG LOSS: {}\n" \
           "TRAINING AVG DIFF: {}".format(
        checkpoint["epoch_avg_loss"], checkpoint["epoch_avg_diff"])


loss_function = nn.MSELoss()
if TRAIN:
    optimizer = optim.Adam(net.parameters())
else:
    old_model_string = loadModel()

loss_history = [9999999]  # very high loss because loss can't be empty for min()
# h0 = Variable(torch.randn(, 3, 20))
# c0 = Variable(torch.randn(2, 3, 20))

for epoch_idx in range(EPOCHS):

    loss_epoch = 0
    diff_epoch = 0

    for episode_idx, data in enumerate(dataloader):
        x, y = makeIntoVariables(data)

        # use random images before feeding real images
        images = autograd.Variable(
            torch.from_numpy(np.random.rand(x.size()[0], 150, 3, 128, 128).astype('float32')),
            requires_grad=False
        ).cuda()
        # reset hidden lstm units
        net.hidden = net.init_hidden()

        loss_episode = 0
        if TRAIN:
            optimizer.zero_grad()

        output_seq = net(x, images)
        loss_episode = loss_function(output_seq, y)
        loss_episode.backward()
        loss_epi = loss_episode.data.cpu()[0]

        if TRAIN:
            optimizer.step()

        # loss.detach()
        net.hidden[0].detach()
        net.hidden[1].detach()

        diff_episode = torch.sum(torch.pow(x.data - y.data, 2))
        printEpisodeLoss(epoch_idx, episode_idx, loss_epi, diff_episode, len(x))

        loss_epoch += loss_epi
        diff_epoch += diff_episode

    printEpochLoss(epoch_idx, episode_idx, loss_epoch, diff_epoch)
    if TRAIN:
        saveModel(
            state=net.state_dict(),
            epoch=epoch_idx,
            episode_idx=episode_idx,
            loss_epoch=loss_epoch,
            diff_epoch=diff_epoch,
            is_best=(loss_epoch < min(loss_history))
        )
        loss_history.append(loss_epoch)
    else:
        print (old_model_string)
        break
