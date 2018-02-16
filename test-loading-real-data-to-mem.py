from torch.utils.data import DataLoader
from tqdm import tqdm

from simple_joints_lstm.dataset_real import DatasetRealPosVel
import numpy as np

DATASET_PATH = "/windata/sim2real-full/done/"
BATCH_SIZE = 1

dataset_train = DatasetRealPosVel(DATASET_PATH, for_training=True, with_velocities=True)
dataset_test = DatasetRealPosVel(DATASET_PATH, for_training=False, with_velocities=True)

dataloader_train = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=1)
dataloader_test = DataLoader(dataset_test, batch_size=BATCH_SIZE, shuffle=True, num_workers=1)

dataset_in = []
dataset_diff = []
dataset_out = []


def makeIntoVariables(dataslice):
    print(dataslice["state_next_sim_joints"].shape)
    print(dataslice["state_current_real_joints"].shape)
    print(dataslice["action"].shape)
    print(dataslice["state_next_real_joints"].shape)
    # x = (dataslice["state_next_sim_joints"], dataslice["state_current_real_joints"], dataslice["action"])
    # y = dataslice["state_next_real_joints"]


for epi, data in enumerate(tqdm(dataloader_test)):
    # makeIntoVariables(data)
    for i in range(data["state_next_sim_joints"].shape[1]):
        dataset_in.append(
            np.hstack((data["state_next_sim_joints"][0, i, :].numpy(),
                       data["state_current_real_joints"][0, i, :].numpy(),
                       data["action"][0, i, :].numpy())
                      ))
        dataset_out.append(
            data["state_next_real_joints"][0, i, :].numpy()
        )
        dataset_diff.append(
            (data["state_next_real_joints"][0, i, :] - data["state_next_sim_joints"][0, i, :]).numpy()
        )

dataset_in = np.array(dataset_in)
dataset_diff = np.array(dataset_diff)
dataset_out = np.array(dataset_out)

print (dataset_in.shape)
print (dataset_diff.shape)
print (dataset_out.shape)


np.savez("dataset-real-test.npz", ds_in=dataset_in, ds_out=dataset_out, ds_diff=dataset_diff)

