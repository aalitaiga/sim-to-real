import matplotlib

from simple_joints_lstm.mujoco_traintest_dataset_pusher3dof import MujocoTraintestPusher3DofDataset
import matplotlib.pyplot as plt
import numpy as np

ms1d = MujocoTraintestPusher3DofDataset(
        "/media/florian/shapenet/sim2real/mujoco_data_pusher3dof_2_tr-1.0_ts-0.85_act-5.h5",
        for_training=True
    )

SAMPLE = 1

data_slice = ms1d[SAMPLE]
state_next_sim_joints = data_slice["state_next_sim_joints"].numpy()
state_next_real_joints = data_slice["state_next_real_joints"].numpy()

diffs = np.deg2rad(state_next_real_joints[:,:3] - state_next_sim_joints[:,:3])

for i in range(3):
    plt.plot(np.arange(len(diffs[:,0])),diffs[:,i], label="m{}".format(i+1))
plt.title("Joint differences over time (real-sim)")
plt.ylabel("Joint difference in degrees")
plt.xlabel("Frame")
plt.legend()
plt.show()

