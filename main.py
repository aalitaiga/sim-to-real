import shutil

from scipy.misc import imsave
import numpy as np
from torch.autograd import Variable
from torch import nn, optim, torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from cyclegan_pix2pix.options.test_options import TestOptions
from models.models import create_model
# absolute imports here, so that you can run the file directly
from simple_joints_lstm.lstm_simple_net import LstmSimpleNet
from simple_joints_lstm.mujoco_traintest_dataset import MujocoTraintestDataset
from simple_joints_lstm.params_adrien import *

dataset = MujocoTraintestDataset(DATASET_PATH, for_training=TRAIN)

batch_size = 16
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1)

net = LstmSimpleNet(batch_size)
net.cuda()

# Load generative model
# python test.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix
# --which_model_netG unet_256 --which_direction BtoA --dataset_mode aligned --norm batch


opt = TestOptions().parse()
# Change here to 'cyclegan' and 'unaligned' for cyclegan
opt.model = 'pix2pix'
opt.name = 'reacher_pix2pix'
opt.dataset_mode = 'aligned'
opt.dataroot = '/Tmp/alitaiga/sim-to-real/reacher_data/AB'
opt.norm = 'batch'
opt.which_direction = 'AtoB'
opt.which_model_netG = 'unet_128'
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 8  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip

model = create_model(opt)

if TRAIN:
    print("STARTING IN TRAINING MODE")
else:
    print("STARTING IN VALIDATION MODE")


def makeIntoVariables(dataslice):
    x, y, z, zz = [Variable(
        dataslice[k],
        requires_grad=False
    ) for k in [
        "state_next_sim_joints",
        "state_next_real_joints",
        "next_sim_images",
        "next_real_images"
    ]]
    z, zz = z.float(), zz.float()
    if CUDA:
        return x.cuda(), y.cuda(), z.cuda(), zz.cuda()
    return x, y, z, zz

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
transform = transforms.Normalize((0.5, 0.5, 0.5),
                     (0.5, 0.5, 0.5))
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
        x, y, z, _ = [data[k].cuda() for k in [
            "state_next_sim_joints",
            "state_next_real_joints",
            "next_sim_images",
            "next_real_images"
        ]]
        # z = z.float()
        z = z[0,:8,:,:,:]
        # z = z.unsqueeze(0)
        z = transform(z)

        out = model.netG.forward(Variable(z, volatile=True))
        imsave('out.jpg', out.data.cpu().squeeze().permute(2,1,0).numpy())
        import sys; sys.exit(0)

        net.hidden = net.init_hidden()

        loss_episode = 0
        if TRAIN:
            optimizer.zero_grad()

        output_seq = net(x, z)
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
