import torch.nn.functional as F
from torch import nn, autograd, torch

from .params import *

# with 8 prediction channels instead of 4
class LstmSimpleNet3(nn.Module):
    def __init__(self):
        super(LstmSimpleNet3, self).__init__()

        # because the LSTM is looking at 1 element at a time, and each element has 4 values
        self.linear1 = nn.Linear(8, HIDDEN_NODES)
        self.lstm1 = nn.LSTM(HIDDEN_NODES, HIDDEN_NODES, LSTM_LAYERS)
        self.linear2 = nn.Linear(HIDDEN_NODES, 8)

        self.hidden = self.init_hidden()

    def zero_hidden(self):
        # print(dir(self.hidden[0]))
        self.hidden[0].data.zero_()
        self.hidden[1].data.zero_()

    def init_hidden(self):
        # the 1 here in the middle is the minibatch size
        h, c = (autograd.Variable(torch.zeros(LSTM_LAYERS, 1, HIDDEN_NODES), requires_grad=False),
                autograd.Variable(torch.zeros(LSTM_LAYERS, 1, HIDDEN_NODES), requires_grad=False))
        if CUDA:
            return h.cuda(), c.cuda()
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

        # out_pos = F.tanh(out[:,:2])
        # out_vel = out[:,2:]
        #
        # out = torch.cat((out_pos, out_vel),dim=1)

        return out
