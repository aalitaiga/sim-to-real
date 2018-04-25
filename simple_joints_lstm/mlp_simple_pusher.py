import torch.nn.functional as F
from torch import nn, torch
from torch.autograd import Variable

from .params_pusher import *


class LstmSimpleNet2Pusher(nn.Module):
    def __init__(self, n_input=15, n_output=6, cuda=True):
        super(LstmSimpleNet2Pusher, self).__init__()
        self.cuda = cuda

        self.fc1 = nn.Linear(n_input, HIDDEN_NODES)
        self.fc2 = nn.Linear(HIDDEN_NODES, HIDDEN_NODES)
        self.fc3 = nn.Linear(HIDDEN_NODES, n_output)


    def forward(self, data_in):
        out = F.leaky_relu(self.fc1(data_in))
        out = F.leaky_relu(self.fc2(out))
        out = self.fc3(out)
        return out



class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features