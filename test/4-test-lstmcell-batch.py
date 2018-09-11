import numpy as np
import torch
import matplotlib.pyplot as plt
from torch import nn, optim, Tensor
from tqdm import tqdm

HIDDEN_CELLS = 20
EPISODES = 10

x_train = torch.from_numpy(torch.load('data_x.pt')).float()
y_train = torch.from_numpy(torch.load('data_y.pt')).float()

print ("x train dims:",x_train.size())

x_full = torch.load('full_x.pt')
y_full = torch.load('full_y.pt')

# plt.plot(np.linspace(-10, 10, 1000), x_full, label="inputs")
# plt.plot(np.linspace(-10, 10, 1000), y_full, label="outputs (target)")
# plt.legend()
# plt.show()


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.linear1 = nn.Linear(1, HIDDEN_CELLS)
        self.lstm1 = nn.LSTMCell(HIDDEN_CELLS, HIDDEN_CELLS)
        self.linear2 = nn.Linear(HIDDEN_CELLS, 1)

    def forward(self, input):
        # input shape should be (10 /minibatch, 10 /steps)

        outputs = []
        h_t = torch.zeros(input.size(0), HIDDEN_CELLS, dtype=torch.float)
        c_t = torch.zeros(input.size(0), HIDDEN_CELLS, dtype=torch.float)
        # h_t2 = torch.zeros(input.size(0), 5, dtype=torch.double)
        # c_t2 = torch.zeros(input.size(0), 5, dtype=torch.double)

        chunked_in = input.chunk(input.size(1), dim=1) # slice through minibatch

        for i, input_t in enumerate(chunked_in):
            hidden = self.linear1(input_t)
            h_t, c_t = self.lstm1(hidden, (h_t, c_t))
            # h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear2(h_t)
            outputs += [output]
        outputs = torch.stack(outputs, 1).squeeze(2)
        return outputs

net = Net()
criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters())

losses = []

for i in tqdm(range(EPISODES)):
    for data_slice_idx in range(x_train.size(0)):
        def closure():
            optimizer.zero_grad()
            out = net(x_train[data_slice_idx])
            loss = criterion(out, y_train[data_slice_idx])
            losses.append(loss.item())
            loss.backward()
            return loss
        optimizer.step(closure)

plt.plot(range(len(losses)),losses)
plt.show()

test = Tensor(x_full).unsqueeze(0)
out_learned = net(test).squeeze(0).data.numpy()

plt.plot(np.linspace(-10, 10, 1000), x_full, label="inputs")
plt.plot(np.linspace(-10, 10, 1000), y_full, label="outputs (target)")
plt.plot(np.linspace(-10, 10, 1000), out_learned, label="outputs (net)")
plt.legend()
plt.show()




