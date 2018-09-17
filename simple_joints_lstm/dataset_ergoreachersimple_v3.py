import os

from torch.utils.data import Dataset
import numpy as np


class DatasetErgoreachersimpleV3(Dataset):
    def __init__(self, path="~/data/sim2real/data-ergoreachersimple-v3-stepping10.npz", train=True):
        super().__init__()
        ds = np.load(os.path.expanduser(path))

        self.curr_real = ds["state_current_real"]
        self.next_real = ds["state_next_real"]
        self.next_sim = ds["state_next_sim"]
        self.action = ds["actions"]

        if train:
            self.curr_real = self.curr_real[:900]
            self.next_real = self.next_real[:900]
            self.next_sim = self.next_sim[:900]
            self.action = self.action[:900]
        else:
            self.curr_real = self.curr_real[900:]
            self.next_real = self.next_real[900:]
            self.next_sim = self.next_sim[900:]
            self.action = self.action[900:]

    def __len__(self):
        return len(self.curr_real)

    def format_data(self, idx):
        return (
            np.hstack((self.next_sim[idx], self.curr_real[idx], self.action[idx])),
            self.next_real[idx] - self.next_sim[idx]
        )

    def __getitem__(self, idx):
        x, y = self.format_data(idx)
        return {"x": x, "y": y}


if __name__ == '__main__':
    dsr = DatasetErgoreachersimpleV3(train=False)
    print("len test", len(dsr))

    print(dsr[10])

    dsr = DatasetErgoreachersimpleV3(train=True)
    print("len train", len(dsr))

    print(dsr[10]["x"].shape, dsr[10]["y"].shape)

    for i in range(10, 20):
        print("real t1:", dsr[0]["x"][i, 8:16].round(2))
        print("sim_ t2:", dsr[0]["x"][i, :8].round(2))
        print("action_:", dsr[0]["x"][i, 16:].round(2))
        print("real t2:", (dsr[0]["x"][i, :8] + dsr[0]["y"][i]).round(2))
        print("delta__:", dsr[0]["y"][i].round(2))
        print("===")

    max_x = -np.inf
    min_x = +np.inf
    max_y = -np.inf
    min_y = +np.inf

    for item in dsr:
        if item["x"].max() > max_x:
            max_x = item["x"].max()
        if item["y"].max() > max_y:
            max_y = item["y"].max()
        if item["x"].min() > min_x:
            min_x = item["x"].min()
        if item["y"].min() > min_y:
            min_y = item["y"].min()

    print("min x {}, max x {}\n"
          "min y {}, max y {}".format(min_x, max_x, min_y, max_y))
