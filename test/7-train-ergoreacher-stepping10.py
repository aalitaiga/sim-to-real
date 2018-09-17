import numpy as np
import torch
import matplotlib.pyplot as plt
from hyperdash import Experiment
from torch import nn, optim, Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from simple_joints_lstm.dataset_ergoreachersimple_v3 import DatasetErgoreachersimpleV3

HIDDEN_CELLS = 50
EPISODES = 30
BATCH_SIZE = 32
EXPERIMENT = 1

# exp 2, 20 cells
# | test loss:   3.432147 |
# | test diff:  65.292154 |

# exp 3, 40 cells
# | test loss:   2.433643 |
# | test diff:  65.292154 |

# exp 4, 50 cells
# | test loss:  29.001207 |
# | test diff: 204.955687 |

PATH = "../trained_models/lstm_ers_v4_exp{}_l{}_n{}.pt".format(
    EXPERIMENT,
    3,
    HIDDEN_CELLS
)

dataset_train = DatasetErgoreachersimpleV3(train=True)
dataset_test = DatasetErgoreachersimpleV3(train=False)

dataloader_train = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
dataloader_test = DataLoader(dataset_test, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.linear1 = nn.Linear(20, HIDDEN_CELLS)
        self.lstm1 = nn.LSTMCell(HIDDEN_CELLS, HIDDEN_CELLS)
        self.lstm2 = nn.LSTMCell(HIDDEN_CELLS, HIDDEN_CELLS)
        self.lstm3 = nn.LSTMCell(HIDDEN_CELLS, HIDDEN_CELLS)
        self.linear2 = nn.Linear(HIDDEN_CELLS, 8)

    def zero_hidden(self, batch_size):
        self.h_t = torch.zeros(batch_size, HIDDEN_CELLS, dtype=torch.float)
        self.c_t = torch.zeros(batch_size, HIDDEN_CELLS, dtype=torch.float)

        self.h_t2 = torch.zeros(batch_size, HIDDEN_CELLS, dtype=torch.float)
        self.c_t2 = torch.zeros(batch_size, HIDDEN_CELLS, dtype=torch.float)

        self.h_t3 = torch.zeros(batch_size, HIDDEN_CELLS, dtype=torch.float)
        self.c_t3 = torch.zeros(batch_size, HIDDEN_CELLS, dtype=torch.float)


    def forward(self, input):
        # input shape should be (32 /minibatch, 100 /steps, 20 /features)

        outputs = []
        self.zero_hidden(input.size(0))

        chunked_in = input.chunk(input.size(1), dim=1) # slice through minibatch

        for i, input_t in enumerate(chunked_in):
            hidden = self.linear1(input_t).squeeze(1)

            self.h_t, self.c_t = self.lstm1(hidden, (self.h_t, self.c_t))
            self.h_t2, self.c_t2 = self.lstm2(self.h_t, (self.h_t2, self.c_t2))
            self.h_t3, self.c_t2 = self.lstm3(self.h_t2, (self.h_t3, self.c_t3))

            output = self.linear2(self.h_t3)

            outputs += [output]

        outputs = torch.stack(outputs, 1)
        return outputs

    def infer(self, input_t):
        # IMPORTANT: before calling this on a rollout the
        # hidden state must be zeroed via self.zero_hidden(1)

        # input shape should be (20 /features)
        hidden = self.linear1(input_t.view(1,1,20)).squeeze(1)

        self.h_t, self.c_t = self.lstm1(hidden, (self.h_t, self.c_t))
        self.h_t2, self.c_t2 = self.lstm2(self.h_t, (self.h_t2, self.c_t2))
        self.h_t3, self.c_t2 = self.lstm3(self.h_t2, (self.h_t3, self.c_t3))

        output = self.linear2(self.h_t3).squeeze(0)

        return output


net = Net()

criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters())

losses = []

exp = Experiment("[sim2real] lstm-ers v3")
exp.param("hidden", HIDDEN_CELLS)
exp.param("episodes", EPISODES)
exp.param("batch", BATCH_SIZE)
exp.param("experiment", EXPERIMENT)


for ep in range(EPISODES):
    for slice in dataloader_train:
        # print (slice["x"].size())
        # print (slice["y"].size())
        # net(slice["x"])

        def closure():
            optimizer.zero_grad()
            out = net(slice["x"])
            loss = criterion(out, slice["y"])

            losses.append(loss.item())
            exp.metric("episode", ep)
            exp.metric("loss", loss.item())
            exp.metric("diff",(slice["y"].data.numpy() ** 2).mean(axis=None))
            loss.backward()
            return loss

        optimizer.step(closure)

    torch.save(net.state_dict(), PATH)

    # net.load_state_dict(torch.load(PATH))

    test_diffs = 0
    test_losses = 0
    for slice in dataloader_test:

        with torch.no_grad():
            for batch_idx in range(slice["x"].size(0)):
                net.zero_hidden(1)
                for slice_idx in range(slice["x"].size(1)):
                    out = net.infer(slice["x"][batch_idx, slice_idx])
                    test_losses += ((out.data.numpy() - slice["y"][batch_idx, slice_idx].data.numpy()) ** 2).mean(axis=None)
                    test_diffs += (slice["y"][batch_idx, slice_idx].data.numpy() ** 2).mean(axis=None)

    exp.metric("test loss",test_losses)
    exp.metric("test diff",test_diffs)
