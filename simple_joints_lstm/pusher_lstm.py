import torch.nn.functional as F
from torch import nn, torch
from torch.autograd import Variable

from .params_pusher import *


class LstmSimpleNet2Pusher(nn.Module):
    def __init__(self, n_input=15, n_output=6, use_cuda=True, batch=1, hidden_nodes=HIDDEN_NODES,
            lstm_layers=LSTM_LAYERS, normalized=False):
        super(LstmSimpleNet2Pusher, self).__init__()
        self.use_cuda = use_cuda
        self.batch = batch
        self.normalized = normalized
        self.lstm_layers = lstm_layers
        self.hidden_nodes = hidden_nodes

        # because the LSTM is looking at 1 element at a time, and each element has 4 values
        self.linear1 = nn.Linear(n_input, hidden_nodes)
        self.lstm1 = nn.LSTM(hidden_nodes, hidden_nodes, self.lstm_layers)
        self.linear2 = nn.Linear(hidden_nodes, n_output)

        self.hidden = self.init_hidden()
        Tensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
        self.mean = Variable(Tensor([0.00173789, 0.00352129, -0.00427585, 0.05105286, 0.11881274, -0.13443381]))
        self.std = Variable(Tensor([0.01608515, 0.0170644, 0.01075647, 0.46635619, 0.53578401, 0.32062387]))

    def zero_hidden(self):
        # print(dir(self.hidden[0]))
        self.hidden[0].data.zero_()
        self.hidden[1].data.zero_()

    def init_hidden(self):
        # the 1 here in the middle is the minibatch size
        h = torch.zeros(self.lstm_layers, self.batch, self.hidden_nodes)
        c = torch.zeros(self.lstm_layers, self.batch, self.hidden_nodes)

        if self.use_cuda:
            h = h.cuda()
            c = c.cuda()

        h, c = (Variable(h),
                Variable(c))
        return h, c

    def forward(self, data_in):
        out = F.leaky_relu(self.linear1(data_in))
        out, self.hidden = self.lstm1(out.permute((1, 0, 2)), self.hidden)
        out = F.leaky_relu(out.permute((1, 0, 2)))
        out = self.linear2(out)

        if self.normalized:
            return (out * self.std) + self.mean
        else:
            return out
