import os

from s2rr.movements.dataset import DatasetProduction
from torch.utils.data import Dataset
import numpy as np


class DatasetRealSmolBulletV2(Dataset):
    def __init__(self, path="~/data/sim2real/data-realigned-v3-{}-bullet.npz", train=True):
        super().__init__()

        ds_name = "train"
        if not train:
            ds_name = "test"
        path_ext = path.format(ds_name)

        print ("using dataset:",path_ext)

        self.ds = DatasetProduction()
        self.ds.load(path_ext)

    def __len__(self):
        return len(self.ds.current_real)

    def format_data(self, idx):
        return (
            np.hstack((self.ds.next_sim[idx], self.ds.current_real[idx], self.ds.action[idx])),
            self.ds.next_real[idx] - self.ds.next_sim[idx]
        )

    def __getitem__(self, idx):
        x, y = self.format_data(idx)
        return {"x": x, "y": y}


if __name__ == '__main__':
    dsr = DatasetRealSmolBulletV2(train=False)
    print("len test", len(dsr))

    print(dsr[10])

    dsr = DatasetRealSmolBulletV2(train=True)
    print("len train", len(dsr))

    print(dsr[10]["x"].shape,dsr[10]["y"].shape)

    for i in range(10,20):

        print("real t1:", dsr[0]["x"][i,12:24].round(2))
        print("sim_ t2:", dsr[0]["x"][i,:12].round(2))
        print("action_:", dsr[0]["x"][i,24:].round(2))
        print("real t2:", (dsr[0]["x"][i,:12] + dsr[0]["y"][i]).round(2))
        print("delta__:", dsr[0]["y"][i].round(2))
        print("===")
