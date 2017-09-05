import torch.nn.functional as F
from torch import nn, autograd, torch

from .params import *


class LstmSimpleNet(nn.Module):
    def __init__(self, batch_size):
        super(LstmSimpleNet, self).__init__()
        self.batch_size = batch_size

        # because the LSTM is looking at 1 element at a time, and each element has 4 values
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=4, padding=1)
        self.conv2 = nn.Conv2d(16, 4, kernel_size=4, stride=4, padding=1)
        self.lstm1 = nn.LSTM(8*8*4 + 4, HIDDEN_NODES, LSTM_LAYERS, batch_first=True)
        self.linear = nn.Linear(HIDDEN_NODES, 4)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        # the 1 here in the middle is the minibatch size
        h, c = (autograd.Variable(torch.zeros(LSTM_LAYERS, 1, HIDDEN_NODES), requires_grad=False),
                autograd.Variable(torch.zeros(LSTM_LAYERS, 1, HIDDEN_NODES), requires_grad=False))
        if CUDA:
            return h.cuda(), c.cuda()
        return h, c

    def forward(self, joint_data, images):
        images = images.view(-1, 3, 128, 128)
        img_features = F.leaky_relu(self.conv1(images))
        img_features = F.leaky_relu(self.conv2(img_features)).view(-1, 256)
        img_features = img_features.view(-1, 150, 256)

        lstm_input = torch.cat((joint_data, img_features), dim=2)
        out, h = self.lstm1(lstm_input)
        out = self.linear(out)
        out_pos = F.tanh(out[:,:,:2])
        out_vel = out[:,:,2:]

        return torch.cat((out_pos, out_vel), dim=2)
