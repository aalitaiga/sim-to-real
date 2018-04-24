import torch.nn.functional as F
from torch import nn, torch
from torch.autograd import Variable

from .params_pusher import *


class LstmReacher(nn.Module):
    def __init__(self, n_input=15, n_output=6, use_cuda=True, batch=1, hidden_nodes=HIDDEN_NODES, normalized=False):
        super(LstmReacher, self).__init__()
        self.use_cuda = use_cuda
        self.batch = batch
        self.normalized = normalized
        self.hidden_nodes = hidden_nodes

        # because the LSTM is looking at 1 element at a time, and each element has 4 values
        self.linear1 = nn.Linear(n_input, hidden_nodes)
        self.lstm1 = nn.LSTM(hidden_nodes, hidden_nodes, LSTM_LAYERS)
        self.linear2 = nn.Linear(hidden_nodes, n_output)
        self.sigmoid = nn.Sigmoid()

        self.hidden = self.init_hidden()
        
    def zero_hidden(self):
        # print(dir(self.hidden[0]))
        self.hidden[0].data.zero_()
        self.hidden[1].data.zero_()

    def init_hidden(self):
        # the 1 here in the middle is the minibatch size
        h = torch.zeros(LSTM_LAYERS, self.batch, self.hidden_nodes)
        c = torch.zeros(LSTM_LAYERS, self.batch, self.hidden_nodes)

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

        return out
