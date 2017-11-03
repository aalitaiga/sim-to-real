import torch
from fuel.datasets import H5PYDataset
from torch.utils.data import Dataset
import h5py

class MujocoTraintestDataset(Dataset):
    """loads the mujoco h5 recording file and makes it available for pytorch training"""

    def __init__(self, h5_file, for_training=True):
        """
        Args:
            h5_file (string): Path to the h5 file
            for_training (bool): True if you want the training dataset, otherwise you get the testing split
        """
        super(MujocoTraintestDataset, self).__init__()
        self.f = h5py.File(h5_file, "r")
        phase = "train"
        if not for_training:
            phase = "valid"
        self.f = H5PYDataset(h5_file, which_sets=(phase,))

    def __len__(self):
        return self.f.num_examples

    def __getitem__(self, idx):
        handle = self.f.open()
        data = self.f.get_data(handle, slice(idx, idx + 1))

        # items:
        # 0 - cos(joint 1 angle)
        # 1 - cos(joint 2 angle)
        # 2 - sin(joint 1 angle)
        # 3 - sin(joint 2 angle)
        # 4 - constant
        # 5 - constant
        # 6 - joint 1 velocity / angular momentum
        # 7 - joint 2 velocity / angular momentum
        # 8-10 distance fingertip to reward object

        relevant_items = [2, 3, 6, 7]  # both angles and velocities
        episode = {  # 'state_joints': torch.from_numpy(data[2][0][:, relevant_items]),
            # 'state_img': self._totensor(data[1][0]),
            # 'action': torch.from_numpy(data[0][0]),
            'state_next_sim_joints': torch.from_numpy(data[8][0][:, relevant_items]),
            # 'state_next_sim_img': self._totensor(data[7][0]),
            'state_next_real_joints': torch.from_numpy(data[4][0][:, relevant_items])
            # 'state_next_real_img': self._totensor(data[3][0])
        }

        self.f.close(handle)

        return episode


if __name__ == '__main__':
    ms1d = MujocoTraintestDataset("../mujoco_data.h5")
    print("loaded dataset with {} episodes".format(len(ms1d)))
    sample = ms1d[0]
    state_next_sim_joints = sample["state_next_sim_joints"]
    state_next_real_joints = sample["state_next_real_joints"]

    print(state_next_sim_joints.size())
    print(state_next_real_joints.size())

    print(state_next_sim_joints[100:103])
    print(state_next_real_joints[100:103])
