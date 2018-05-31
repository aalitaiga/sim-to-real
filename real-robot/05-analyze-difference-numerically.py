import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib

matplotlib.style.use('ggplot')

ds = np.load(os.path.expanduser("~/data/sim2real/dataset-real-{}-normalized-bullet.npz".format("test")))

ds_curr_real = ds["ds_curr_real"]
ds_next_real = ds["ds_next_real"]
ds_next_sim = ds["ds_next_sim"]
ds_action = ds["ds_action"]
ds_epi = ds["ds_epi"]

action_data = []
joint1_data = []

for sample_idx in range(10):
    epi = np.random.randint(0, ds_epi.max())

    # print(ds_curr_real[ds_epi == epi].shape)

    action_data.append(ds_action[ds_epi == epi][0:30, 0])
    joint1_data.append(ds_curr_real[ds_epi == epi][0:30, 0])

    # for i in range(50):
    # print (i,"=")
    # print("real t1:", ds_curr_real[ds_epi == epi][i].round(2))
    # print("sim_ t2:", ds_next_sim[ds_epi == epi][i].round(2))
    # print("action_:", ds_action[ds_epi == epi][i].round(2))
    # print("real t2:", ds_next_real[ds_epi == epi][i].round(2))
    # print("===")


y_action = np.hstack(action_data)
y_joint1 = np.hstack(joint1_data)
x = np.arange(0, len(y_action))

plt.plot(x, y_joint1, label="joint1")
plt.plot(x, y_action, label="action1")
plt.axhline(0, c="g")
plt.legend()
plt.show()
