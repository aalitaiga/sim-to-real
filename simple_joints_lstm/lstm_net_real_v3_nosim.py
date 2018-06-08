import torch.nn.functional as F
from torch import nn, autograd, torch

from .params_pusher import *


class LstmNetRealv3NoSim(nn.Module):
    def __init__(self, n_input_state_real=12, n_input_actions=6, nodes=128, layers=3):
        super().__init__()

        self.nodes = nodes
        self.layers = layers

        self.linear1 = nn.Linear(n_input_state_real + n_input_actions, nodes)
        self.lstm1 = nn.LSTM(nodes, nodes, layers)
        self.linear2 = nn.Linear(nodes, n_input_state_real)

        self.hidden = self.init_hidden()

    def zero_hidden(self):
        self.hidden[0].data.zero_()
        self.hidden[1].data.zero_()

    def init_hidden(self):
        # the 1 here in the middle is the minibatch size
        h = autograd.Variable(torch.zeros(self.layers, 1, self.nodes))
        c = autograd.Variable(torch.zeros(self.layers, 1, self.nodes))

        if torch.cuda.is_available():
            h = h.cuda()
            c = c.cuda()

        return h, c

    def forward(self, data_in):
        out = F.leaky_relu(self.linear1(data_in))
        out, self.hidden = self.lstm1(out, self.hidden)
        out = F.leaky_relu(out)
        out = F.tanh(self.linear2(out))*2 # added tanh for normalization
        return out
