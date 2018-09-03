import torch.nn.functional as F
from torch import nn, autograd, torch

from .params_pusher import *


# same as v3 but output isn't multiplied by 2
# because upon inspecting the dataset y is always <0.85
# ... also converted to torch v0.4
class LstmNetRealv5(nn.Module):
    def __init__(
            self,
            n_input_state_sim=12,
            n_input_state_real=12,
            n_input_actions=6,
            nodes=128,
            layers=3,
            cuda=True
    ):
        super().__init__()

        self.with_cuda = cuda

        self.nodes = nodes
        self.layers = layers

        self.linear1 = nn.Linear(n_input_state_sim + n_input_state_real + n_input_actions, nodes)
        self.lstm1 = nn.LSTM(nodes, nodes, layers)
        self.linear2 = nn.Linear(nodes, n_input_state_real)

        self.hidden = self.init_hidden()

        self.leaky_relu = torch.nn.LeakyReLU()

    def zero_hidden(self):
        self.hidden[0].data.zero_()
        self.hidden[1].data.zero_()

    def init_hidden(self):
        # the 1 here in the middle is the minibatch size
        h = autograd.Variable(torch.zeros(self.layers, 1, self.nodes))
        c = autograd.Variable(torch.zeros(self.layers, 1, self.nodes))

        if torch.cuda.is_available() and self.with_cuda:
            h = h.cuda()
            c = c.cuda()

        return h, c

    def detach_hidden(self):
        self.hidden = (self.hidden[0].detach(), self.hidden[1].detach())

    def forward(self, data_in):
        out = self.leaky_relu(self.linear1(data_in))
        out, self.hidden = self.lstm1(out, self.hidden)
        out = self.leaky_relu(out)
        out = torch.tanh(self.linear2(out))  # added tanh for normalization
        return out
