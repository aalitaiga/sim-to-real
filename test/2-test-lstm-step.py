import copy

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn, Tensor
from tqdm import tqdm

EPISODES = 10000


class LstmNet(nn.Module):
    def __init__(
            self,
            n_input=1,
            nodes=5,
            layers=5,
            cuda=True
    ):
        super().__init__()

        self.with_cuda = cuda

        self.nodes = nodes
        self.layers = layers

        self.linear1 = nn.Linear(n_input, nodes)
        self.lstm1 = nn.LSTM(nodes, nodes, layers)
        self.linear2 = nn.Linear(nodes, n_input)

        self.hidden = self.init_hidden()

        self.leaky_relu = torch.nn.LeakyReLU()

    def zero_hidden(self):
        self.hidden[0].data.zero_()
        self.hidden[1].data.zero_()

    def init_hidden(self):
        # the 1 here in the middle is the minibatch size
        h = torch.zeros(self.layers, 1, self.nodes)
        c = torch.zeros(self.layers, 1, self.nodes)

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
        out = torch.tanh(self.linear2(out))
        return out


space = np.linspace(-10, 10, 1000)
x = np.sin(space)
y = np.sin(space + 1)
losses = []
model = LstmNet(cuda=False)



def sample(datasetX, datasetY, samples=10):
    ds_len = len(datasetX) - samples
    start = np.random.randint(0, ds_len)
    return datasetX[start:start + 10], datasetY[start:start + 10]


loss_function = nn.MSELoss()

optimizer = torch.optim.Adam(model.parameters())
losses = []

for ep in tqdm(range(EPISODES)):
    model.detach_hidden()
    model.zero_hidden()

    dx, dy = sample(x, y)
    loss = torch.zeros(1,dtype=torch.float)
    for idx in range(len(dx)):
        dxe = Tensor([dx[idx]]).view(1,1,1)
        dye = Tensor([dy[idx]]).view(1,1,1)
        result = model.forward(dxe)
        loss += loss_function(result, dye)
        losses.append(loss.data.numpy())

    loss.backward()
    optimizer.step()

plt.plot(range(len(losses)), losses)
plt.show()


dxv = Tensor(x).view(-1, 1, 1)
result = model.forward(dxv).view(-1).data.numpy()

plt.plot(space, x, label="x")
plt.plot(space, y, label="y_true")
plt.plot(space, result, label="y_pred")
plt.title("step")
plt.legend()
plt.show()
