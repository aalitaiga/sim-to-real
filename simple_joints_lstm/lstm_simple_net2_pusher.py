import torch.nn.functional as F
from torch import nn, autograd, torch

from .params_pusher import *


class LstmSimpleNet2Pusher(nn.Module):
    def __init__(self, n_input=15, n_output=6):
        super(LstmSimpleNet2Pusher, self).__init__()

        # because the LSTM is looking at 1 element at a time, and each element has 4 values
        self.linear1 = nn.Linear(n_input, HIDDEN_NODES)
        self.lstm1 = nn.LSTM(HIDDEN_NODES, HIDDEN_NODES, LSTM_LAYERS)
        self.linear2 = nn.Linear(HIDDEN_NODES, n_output)

        self.hidden = self.init_hidden()

    def zero_hidden(self):
        # print(dir(self.hidden[0]))
        self.hidden[0].data.zero_()
        self.hidden[1].data.zero_()

    def init_hidden(self):
        # the 1 here in the middle is the minibatch size
        h, c = (autograd.Variable(torch.zeros(LSTM_LAYERS, 1, HIDDEN_NODES).cuda(), requires_grad=False),
                autograd.Variable(torch.zeros(LSTM_LAYERS, 1, HIDDEN_NODES).cuda(), requires_grad=False))
        # if CUDA:
        #     return h.cuda(), c.cuda()
        return h, c

    def forward(self, data_in):
        # the view is to add the minibatch dimension (which is 1)
        # print(data_in.view(1, 1, -1).size())

        out = F.leaky_relu(self.linear1(data_in))
        out, self.hidden = self.lstm1(out.permute((1,0,2)), self.hidden)
        out = F.leaky_relu(out.permute((1,0,2)))
        out = self.linear2(out)
        return out
