import os
from torch.utils.data import Dataset
import numpy as np


class DatasetRealSmolBullet(Dataset):
    def __init__(self, path="~/data/sim2real/data-realigned-{}-bullet3.npz", train=True):
        super().__init__()

        ds_name = "train"
        if not train:
            ds_name = "test"
        path_ext = path.format(ds_name)

        print ("using dataset:",path_ext)

        ds = np.load(os.path.expanduser(path_ext))

        self.curr_real = ds["ds_curr_real"]
        self.next_real = ds["ds_next_real"]
        self.next_sim = ds["ds_next_sim"]
        self.action = ds["ds_action"]
        self.epi = ds["ds_epi"]

    def __len__(self):
        return len(self.curr_real)

    def format_data(self, idx):
        return (
            np.hstack((self.next_sim[idx], self.curr_real[idx], self.action[idx])),
            self.next_real[idx] - self.next_sim[idx],
            self.epi[idx]
        )

    def __getitem__(self, idx):
        x, y, epi = self.format_data(idx)
        return {"x": x, "y": y, "epi": epi}


if __name__ == '__main__':
    dsr = DatasetRealSmolBullet(train=False)
    print("len test", len(dsr))

    print(dsr[10])

    dsr = DatasetRealSmolBullet(train=True)
    print("len train", len(dsr))

    print(dsr[10])

    for i in range(10,20):
        print("real t1:", dsr[i]["x"][12:24].round(2))
        print("sim_ t2:", dsr[i]["x"][:12].round(2))
        print("action_:", dsr[i]["x"][24:].round(2))
        print("real t2:", (dsr[i]["x"][:12] + dsr[i]["y"]).round(2))
        print("===")
