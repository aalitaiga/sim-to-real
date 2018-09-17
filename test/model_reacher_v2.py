import torch
from torch import nn


class ReacherModelV2(nn.Module):
    def __init__(self, hidden_cells):
        super(ReacherModelV2, self).__init__()
        self.hidden = hidden_cells

        self.linear1 = nn.Linear(20, self.hidden)
        self.lstm1 = nn.LSTMCell(self.hidden, self.hidden)
        self.lstm2 = nn.LSTMCell(self.hidden, self.hidden)
        self.lstm3 = nn.LSTMCell(self.hidden, self.hidden)
        self.linear2 = nn.Linear(self.hidden, 8)

    def zero_hidden(self, batch_size):
        self.h_t = torch.zeros(batch_size, self.hidden, dtype=torch.float)
        self.c_t = torch.zeros(batch_size, self.hidden, dtype=torch.float)

        self.h_t2 = torch.zeros(batch_size, self.hidden, dtype=torch.float)
        self.c_t2 = torch.zeros(batch_size, self.hidden, dtype=torch.float)

        self.h_t3 = torch.zeros(batch_size, self.hidden, dtype=torch.float)
        self.c_t3 = torch.zeros(batch_size, self.hidden, dtype=torch.float)


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
        # which consists of
        # 8 - sim_{t+1}
        # 8 - real_{t}
        # 4 - action
        hidden = self.linear1(input_t.view(1,1,20)).squeeze(1)

        self.h_t, self.c_t = self.lstm1(hidden, (self.h_t, self.c_t))
        self.h_t2, self.c_t2 = self.lstm2(self.h_t, (self.h_t2, self.c_t2))
        self.h_t3, self.c_t2 = self.lstm3(self.h_t2, (self.h_t3, self.c_t3))

        output = self.linear2(self.h_t3).squeeze(0)

        return output
