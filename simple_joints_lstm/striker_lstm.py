import torch.nn.functional as F
from torch import nn, torch
from torchqrnn import QRNN
from torch.autograd import Variable

from simple_joints_lstm.weight_drop import WeightDrop
from .params_pusher import *


class LstmStriker(nn.Module):
    def __init__(self, n_input=15, n_output=6, use_cuda=True, batch=1, hidden_nodes=HIDDEN_NODES,
            lstm_layers=LSTM_LAYERS, use_qrnn=False, wdrop=0., dropouti=0.):
        super(LstmStriker, self).__init__()
        self.use_cuda = use_cuda
        self.batch = batch
        # self.idrop = nn.Dropout(dropouti)
        self.odrop = nn.Dropout(dropouti)
        self.lstm_layers = lstm_layers
        self.hidden_nodes = hidden_nodes

        self.linear1 = nn.Linear(n_input, hidden_nodes)
        # self.batch_norm = nn.BatchNorm1d(hidden_nodes)
        if use_qrnn:
            self.lstm1 = QRNN(hidden_nodes, hidden_nodes, num_layers=LSTM_LAYERS, dropout=0.4)
        else:
            self.lstm1 = nn.LSTM(hidden_nodes, hidden_nodes, self.lstm_layers)
            if wdrop:
                self.lstm1 = WeightDrop(self.lstm1, ['weight_hh_l0'], dropout=wdrop)
        self.linear2 = nn.Linear(hidden_nodes, n_output)

        self.hidden = self.init_hidden()

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
        # out = self.idrop(out)
        out, self.hidden = self.lstm1(out.permute((1, 0, 2)), self.hidden)
        out = F.leaky_relu(out.permute((1, 0, 2)))
        out = self.odrop(out)
        out = self.linear2(out)

        return out

class LstmStriker2(nn.Module):
    def __init__(self, n_input=15, n_output=6, use_cuda=True, batch=1, hidden_nodes=HIDDEN_NODES,
            lstm_layers=LSTM_LAYERS, use_qrnn=False, wdrop=0.5, dropouti=0.3):
        super(LstmStriker, self).__init__()
        self.use_cuda = use_cuda
        self.batch = batch

        self.lstm_cell = lstm_layers
        self.hidden_nodes = hidden_nodes

        self.linear1 = nn.Linear(n_input, hidden_nodes)
        self.lstm1 = nn.LSTM(hidden_nodes, hidden_nodes, self.lstm_layers)
        if wdrop:
            self.lstm1 = WeightDrop(self.lstm1, ['weight_hh_l0'], dropout=wdrop)
        self.linear2 = nn.Linear(hidden_nodes, n_output)

        self.hidden = self.init_hidden()

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
        # out = F.leaky_relu(self.batch_norm(self.linear1(data_in).view(-1, 256))).view(self.batch, -1, self.hidden_nodes)
        out = F.leaky_relu(self.linear1(data_in))
        # out = self.idrop(out)
        out, self.hidden = self.lstm1(out.permute((1, 0, 2)), self.hidden)
        out = F.leaky_relu(out.permute((1, 0, 2)))
        # out = self.odrop(out)
        out = self.linear2(out)

        return out

def wrap_zoneout_cell(cell_func, zoneout_prob=0):
    def f(*kargs, **kwargs):
        return ZoneOutCell(cell_func(*kargs, **kwargs), zoneout_prob)
    return f


class ZoneOutCell(nn.Module):

    def __init__(self, cell, zoneout_prob=0):
        super(ZoneOutCell, self).__init__()
        self.cell = cell
        self.hidden_size = cell.hidden_size
        self.zoneout_prob = zoneout_prob

    def forward(self, inputs, hidden):
        def zoneout(h, next_h, prob):
            if isinstance(h, tuple):
                num_h = len(h)
                if not isinstance(prob, tuple):
                    prob = tuple([prob] * num_h)
                return tuple([zoneout(h[i], next_h[i], prob[i]) for i in range(num_h)])
            mask = Variable(h.data.new(h.size()).bernoulli_(
                prob), requires_grad=False)
            return mask * next_h + (1 - mask) * h

        next_hidden = self.cell(inputs, hidden)
        next_hidden = zoneout(hidden, next_hidden, self.zoneout_prob)
        return next_hidden
