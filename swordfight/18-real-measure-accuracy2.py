import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
from torch.autograd import Variable
from tqdm import tqdm

from simple_joints_lstm.lstm_net_real_v3 import LstmNetRealv3
from simple_joints_lstm.lstm_simple_net2_real import LstmSimpleNet2Real

DATASET_PATH = "~/data/sim2real/data-realigned-{}-bullet3.npz"  # {} is either "train" or "test"

ds_train = np.load(os.path.expanduser(DATASET_PATH.format("test")))

x_train = np.hstack([ds_train["ds_next_sim"], ds_train["ds_curr_real"], ds_train["ds_action"]])
y_train = ds_train["ds_next_real"] - ds_train["ds_next_sim"]

x_train = x_train.astype(np.float32)
y_train = y_train.astype(np.float32)

print(x_train.shape)
print(y_train.shape)


def double_unsqueeze(data):
    return torch.unsqueeze(torch.unsqueeze(torch.from_numpy(data), dim=0), dim=0)


def double_squeeze(data):
    return torch.squeeze(torch.squeeze(data)).data.numpy()


def data_to_var(data):
    return (Variable(double_unsqueeze(data)))


def se(A, B):
    return ((A - B) ** 2)


out = []

# ["1_v1_3l_128", "2_v1_3l_256", "3_v1_5l_128", "4_v1_5l_256"]

for model in [(1, 3, 128), (2, 5, 128), (3, 3, 256), (4, 5, 256)]:

    model_file = "../trained_models/lstm_real_v4_exp{}_l{}_n{}_best.pt".format(*model)

    print(model_file)

    stored_model = torch.load(model_file, map_location='cpu')

    lstm = LstmNetRealv3(layers=model[1], nodes=model[2]).float()
    lstm.load_state_dict(stored_model["state_dict"])
    lstm.eval()

    cumm_err = np.zeros(12, dtype=np.float32)

    old_action = x_train[0, 24:].copy()
    action = 0

    for i in tqdm(range(len(x_train))):
        inf = double_squeeze(lstm.forward(data_to_var(x_train[i])))
        print(i, "=")
        print("real t1_x:", np.around(x_train[i, 12:24], 2))
        print("sim_ t2_x:", np.around(x_train[i, :12], 2))
        print("action__x:", np.around(x_train[i, 24:], 2))
        print("real t2_x:", np.around(x_train[i, :12] + y_train[i], 2))
        print("real t2_y:", np.around(x_train[i, :12] + inf, 2))
        print("delta___x:", np.around(y_train[i], 2))
        print("delta___y:", np.around(inf, 2))
        print("===")
        serr = se(inf, y_train[i])
        cumm_err += serr

        if not (old_action == x_train[i, 24:]).all():
            action += 1
            old_action = x_train[i, 24:]

            if action == 3:
                lstm.zero_hidden()
                action = 0

        if i == 20: break

    out.append((model, cumm_err))

for model, err in out:
    print(model, err.mean())

## NORMAL MODELS

# (1, 3, 128) 9419.109
# (2, 5, 128) 9421.687
# (3, 3, 256) 9515.12
# (4, 5, 256) 9308.303

## "BEST" MODELS

# same
