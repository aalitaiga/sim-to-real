from torch.utils.data import Dataset
import numpy as np

from simple_joints_lstm.utils import normalizePusher3Dof

DATASET_SIZE = 1000
VALIDATION_SIZE = 100


class MujocoTraintestPusherSimpleDataset(Dataset):
    """loads the mujoco npz recording file and makes it available for pytorch training"""

    def __init__(self, npz_file, for_training=True):
        """
        Args:
            npz_file (string): Path to the numpy file
            for_training (bool): True if you want the training dataset, otherwise you get the testing split
        """
        super(MujocoTraintestPusherSimpleDataset, self).__init__()
        self.for_training = for_training

        self.f = np.load(npz_file)

    def __len__(self):
        if not self.for_training:
            return VALIDATION_SIZE
        else:
            return DATASET_SIZE - VALIDATION_SIZE

    def __getitem__(self, idx):
        # items:
        # 0,1,2 - joint angles
        # 3,4,5 - joint velocities
        # 6,7 - tip x/y coordinates
        # 8,9 - object x/y coords
        # 10,11 - goal x/y coords
        # relevant_items = range(6)  # both angles and velocities, no coords

        if self.for_training:
            # if training then skip the first N elements
            # (because they are reserved for validation)
            idx += VALIDATION_SIZE

        episode = {
            'state_current_real': normalizePusher3Dof(self.f["state_current_real"][idx, :, :6]),
            'state_next_sim': normalizePusher3Dof(self.f["state_next_sim"][idx, :, :6]),
            'state_next_real': normalizePusher3Dof(self.f["state_next_real"][idx, :, :6]),
            'actions': self.f["actions"][idx, :, :]
        }

        return episode


if __name__ == '__main__':
    ms1d = MujocoTraintestPusherSimpleDataset("../data-collection/mujoco_pusher3dof_simple_1act.npz",
                                              for_training=False)
    print("loaded dataset with {} episodes".format(len(ms1d)))
    sample = ms1d[0]
    state_current_real = sample["state_current_real"]
    state_next_real = sample["state_next_real"]
    state_next_sim = sample["state_next_sim"]
    actions = sample["actions"]

    print(state_current_real.size())
    print(state_next_real.size())
    print(state_next_sim.size())
    print(actions.size())

    print(state_current_real[50:52])
    print(state_next_real[50:52])
    print(state_next_sim[50:52])
    print(actions[50:52])

    max_pos = -np.inf
    min_pos = np.inf
    max_vel = -np.inf
    min_vel = np.inf
    min_act = np.inf
    max_act = -np.inf
    for i in range(len(ms1d)):
        sample = ms1d[0]
        state_current_real = sample["state_current_real"].numpy()
        state_next_real = sample["state_next_real"].numpy()
        state_next_sim = sample["state_next_sim"].numpy()
        actions = sample["actions"].numpy()

        if state_current_real[:, :3].max() > max_pos:
            max_pos = state_current_real[:, :3].max()
        if state_current_real[:, :3].min() < min_pos:
            min_pos = state_current_real[:, :3].min()

        if state_current_real[:, 3:].max() > max_vel:
            max_vel = state_current_real[:, 3:].max()
        if state_current_real[:, 3:].min() < min_vel:
            min_vel = state_current_real[:, 3:].min()

        if actions.max() > max_act:
            max_act = actions.max()
        if actions.min() < min_act:
            min_act = actions.min()

    print("\nmax_pos:", max_pos,
          "\nmin_pos:", min_pos,
          "\nmax_vel:", max_vel,
          "\nmin_vel:", min_vel,
          "\nmin_act:", min_act,
          "\nmax_act:", max_act)

    # output:
    # max_pos: 2.5162039
    # min_pos: -0.1608184
    # max_vel: 12.24464
    # min_vel: -2.2767675
    # min_act: -0.99060905
    # max_act: 0.997694
