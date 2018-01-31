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
        if CUDA:
            h, c = (autograd.Variable(torch.zeros(LSTM_LAYERS, 1, HIDDEN_NODES).cuda(), requires_grad=False),
                    autograd.Variable(torch.zeros(LSTM_LAYERS, 1, HIDDEN_NODES).cuda(), requires_grad=False))
        else:
            h, c = (autograd.Variable(torch.zeros(LSTM_LAYERS, 1, HIDDEN_NODES), requires_grad=False),
                    autograd.Variable(torch.zeros(LSTM_LAYERS, 1, HIDDEN_NODES), requires_grad=False))
        return h, c

    def zero_hidden(self):
        self.hidden[0].data.zero_()
        self.hidden[1].data.zero_()

    def forward(self, data_in):
        # the view is to add the minibatch dimension (which is 1)
        # print(data_in.view(1, 1, -1).size())
        out = F.leaky_relu(self.linear1(data_in.view(1, -1)))
        out, self.hidden = self.lstm1(out.view(1, 1, -1), self.hidden)
        out = F.leaky_relu(out)
        out = self.linear2(out.view(1, -1))

        out_pos = F.tanh(out[:,:2])
        out_vel = out[:,2:]

        lstm_input = torch.cat((joint_data, img_features), dim=2)
        out, h = self.lstm1(lstm_input)
        out = self.linear(out)
        out_pos = F.tanh(out[:,:,:2])
        out_vel = out[:,:,2:]

        return torch.cat((out_pos, out_vel), dim=2)
