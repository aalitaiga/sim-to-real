from torch.utils.data import DataLoader
from tqdm import tqdm

from simple_joints_lstm.dataset_real import DatasetRealPosVel
import numpy as np

DATASET_PATH = "/windata/sim2real-full/done/"
BATCH_SIZE = 1

dataset_train = DatasetRealPosVel(DATASET_PATH, for_training=True, with_velocities=True, normalized=True)
dataset_test = DatasetRealPosVel(DATASET_PATH, for_training=False, with_velocities=True, normalized=True)

dataloader_train = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
dataloader_test = DataLoader(dataset_test, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

for run in ["train", "test"]:
    print("run:", run)

    dataloader = dataloader_train
    if run == "test":
        dataloader = dataloader_test

    dataset_curr_real = []
    dataset_next_real = []
    dataset_action = []
    dataset_epi = []

    for epi, data in enumerate(tqdm(dataloader)):
        for i in range(data["state_next_sim_joints"].shape[1]):
            dataset_curr_real.append(
                data["state_current_real_joints"][0, i, :].numpy()
            )
            dataset_next_real.append(
                data["state_next_real_joints"][0, i, :].numpy()
            )
            dataset_action.append(
                data["action"][0, i, :].numpy()
            )
            dataset_epi.append(epi)

    dataset_curr_real = np.array(dataset_curr_real)
    dataset_next_real = np.array(dataset_next_real)
    dataset_action = np.array(dataset_action)
    dataset_epi = np.array(dataset_epi)

    print(dataset_curr_real.shape)
    print(dataset_next_real.shape)
    print(dataset_action.shape)
    print(dataset_epi.shape)

    np.savez(
        "dataset-real-{}-normalized-onlyRobot.npz".format(run),
        ds_curr_real=dataset_curr_real,
        ds_next_real=dataset_next_real,
        ds_action=dataset_action,
        ds_epi=dataset_epi
    )
