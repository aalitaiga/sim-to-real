from torch.utils.data import Dataset
import h5py

class MujocoSimple1Dataset(Dataset):
    """loads the mujoco h5 recording file and makes it available for pytorch training"""

    def __init__(self, h5_file):
        """
        Args:
            h5_file (string): Path to the h5 file
        """
        self.f = h5py.File(h5_file, "r")

    # def __del__(self): # this actually causes a nasty exception in Py3.5
    #     self.f.close()

    def __len__(self):
        return len(self.f["actions"])

    def __getitem__(self, idx):
        episode = {'state_joints': self.f["obs"][idx],
                  # 'state_img': 0,
                  'action': self.f["actions"][idx],
                  'state_next_sim_joints': self.f["s_transition_obs"][idx],
                  # 'state_next_sim_img':0,
                  'state_next_real_joints': self.f["r_transition_obs"][idx],
                  # 'state_next_real_img': 0
                  }

        return episode

if __name__ == '__main__':
    ms1d = MujocoSimple1Dataset("../mujoco_data.h5")
    print ("loaded dataset with {} episodes".format(len(ms1d)))
    sample = ms1d[0]
    state = sample["state_joints"]
    action = sample["action"]
    state_next_sim_joints = sample["state_next_sim_joints"]
    state_next_real_joints = sample["state_next_real_joints"]

    print (state.shape)
    print (action.shape)
    print (state_next_sim_joints.shape)
    print (state_next_real_joints.shape)

    # print(state[30:39])
    # print(state_next_sim_joints[30:39])
    # print(state_next_real_joints[30:39])
    print(state[100:110])
    print(state_next_sim_joints[100:110])
    print(state_next_real_joints[100:110])
