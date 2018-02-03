import math
import torch
from fuel.datasets import H5PYDataset
from torch.utils.data import Dataset
import h5py
import numpy as np


class DatasetRealPosVel(Dataset):
    """loads the rea h5 recording file (but only pos, vel, actions) and makes it available for pytorch training"""

    def __init__(self, path_prefix, for_training=True, with_velocities=True):
        """
        Args:
            h5_file (string): Path to the h5 file
            for_training (bool): True if you want the training dataset, otherwise you get the testing split
        """
        super(DatasetRealPosVel, self).__init__()
        self.for_training = for_training
        self.path_prefix = path_prefix
        self.with_velocities = with_velocities

        self.fn_pattern = "dataset-{}-{}.hdf5"
        self.open_file_names = []
        self.open_file_handles = []

        ## dataset has 1490 entries
        ## out of which 0-199 are for validation
        ## and 200-1490 are for training

        self.data_range = range(200, 1490)
        if not for_training:
            self.data_range = range(0, 200)

    def __len__(self):
        return len(self.data_range)

    def __getitem__(self, idx):

        ## Each HDF5 file contains 100 episodes.
        ## These 100 episodes are themselves in groups of 10.
        ## Each group of 10 episodes is a "file", and this "file"
        ## constitutes the filename of the HDF5 file.
        ## e.g. "dataset-50-60.hdf5" contains episodes 500-599,
        ## which are split into "file" 50-59, with 10 episodes each.
        ## This artifical "file" group is important when indexing
        ## the episodes within the HDF5 file.

        ## find the name of the dataset to open
        file_lower = int(math.floor(idx / 100) * 10)
        if self.for_training:
            file_lower += 20  # +20 is from train/test offset
        file_upper = file_lower + 10
        if file_lower == 140:
            file_upper = 149

        ## see if we already have this file open. If not, put it on the stack
        file_to_open = self.fn_pattern.format(file_lower, file_upper)
        if file_to_open in self.open_file_names:
            file_handle = self.open_file_handles[self.open_file_names.index(file_to_open)]
        else:
            file_handle = h5py.File(self.path_prefix + file_to_open, "r")
            self.open_file_handles.append(file_handle)
            self.open_file_names.append(file_to_open)

        ## calculate the relative "file" index and episode index of the episode
        ## within the HDF5 file.
        file_idx = int(math.floor(idx / 10)) - file_lower
        if self.for_training:
            file_idx += 20  # +20 is from train/test offset

        episode_idx = idx % 10

        current_pos_real = file_handle.get("current_pos_real")[file_idx, episode_idx]
        current_vel_real = file_handle.get("current_vel_real")[file_idx, episode_idx]
        next_pos_sim = file_handle.get("next_pos_sim")[file_idx, episode_idx]
        next_vel_sim = file_handle.get("next_vel_sim")[file_idx, episode_idx]
        next_pos_real = file_handle.get("next_pos_real")[file_idx, episode_idx]
        next_vel_real = file_handle.get("next_vel_real")[file_idx, episode_idx]

        current_action = file_handle.get("current_action")[file_idx, episode_idx]

        non_zero = np.unique(np.nonzero(current_pos_real)[0], axis=0)
        # print (current_pos_real[non_zero].shape) ## something like (263,6) or (258,6)
        # print (current_vel_real[non_zero].shape) ## the length / number of frames of each episode is different
        # print (next_pos_sim[non_zero].shape)  ## and varies in in [60,273]
        # print (next_vel_sim[non_zero].shape)
        # print (next_pos_real[non_zero].shape)
        # print (next_vel_real[non_zero].shape)
        # print (current_action[non_zero].shape)

        if self.with_velocities:
            current_state_real = np.hstack((current_pos_real[non_zero], current_vel_real[non_zero]))
            next_state_sim = np.hstack((next_pos_sim[non_zero], next_vel_sim[non_zero]))
            next_state_real = np.hstack((next_pos_real[non_zero], next_vel_real[non_zero]))
        else:
            current_state_real = current_pos_real[non_zero]
            next_state_sim = next_pos_sim[non_zero]
            next_state_real = next_pos_real[non_zero]

        episode = {
            'state_current_real_joints': torch.from_numpy(current_state_real),
            'state_next_sim_joints': torch.from_numpy(next_state_sim),
            'state_next_real_joints': torch.from_numpy(next_state_real),
            'action': torch.from_numpy(current_action[non_zero])
        }

        return episode


if __name__ == '__main__':
    dsr = DatasetRealPosVel("/windata/sim2real-full/done/", for_training=False)
    print("len test", len(dsr))

    dsr = DatasetRealPosVel("/windata/sim2real-full/done/", for_training=True)
    print("len train", len(dsr))

    print(dsr[15])

    # print (dsr[15]["state_next_real_joints"][:50])
