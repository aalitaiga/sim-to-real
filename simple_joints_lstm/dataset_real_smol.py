import os
from torch.utils.data import Dataset
import numpy as np


class DatasetRealSmol(Dataset):
    def __init__(self, path="~/data/sim2real/dataset-real-{}-normalized.npz", train=True):
        super().__init__()

        ds_name = "train"
        if not train:
            ds_name = "test"
        path_ext = path.format(ds_name)

        ds = np.load(os.path.expanduser(path_ext))

        self.x = ds["ds_in"]
        self.y = ds["ds_diff"]
        self.episode = ds["ds_epi"]

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return {"x": self.x[idx], "y": self.y[idx], "epi": self.episode[idx]}


if __name__ == '__main__':
    dsr = DatasetRealSmol(train=False)
    print("len test", len(dsr))

    print(dsr[15])

    dsr = DatasetRealSmol(train=True)
    print("len train", len(dsr))

    print(dsr[15])

    for i in range(10):
        print("real t1:", dsr[i]["x"][12:24].round(2))
        print("sim_ t2:", dsr[i]["x"][:12].round(2))
        print("action_:", dsr[i]["x"][24:].round(2))
        print("real t2:", (dsr[i]["x"][:12] + dsr[i]["y"]).round(2))
        print("===")
