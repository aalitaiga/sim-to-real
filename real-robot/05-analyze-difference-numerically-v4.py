import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib
from s2rr.movements.dataset import DatasetProduction

matplotlib.style.use('ggplot')

ds = DatasetProduction()
ds.load("~/data/sim2real/data-realigned-v2-{}-bullet.npz".format("test"))

action_data = []
joint1_data = []

for sample_idx in range(10):
    epi = np.random.randint(0, len(ds.current_real))

    # print(ds_curr_real[ds_epi == epi].shape)

    action_data.append(ds.action[epi, 0:30, 0])
    joint1_data.append(ds.current_real[epi, 0:30, 0])

    for i in range(50):
        print(i, "=")
        print("real t1:", ds.current_real[epi, i].round(2))
        print("sim_ t2:", ds.next_sim[epi, i].round(2))
        print("action_:", ds.action[epi, i].round(2))
        print("real t2:", ds.next_real[epi, i].round(2))
        print("===")

y_action = np.hstack(action_data)
y_joint1 = np.hstack(joint1_data)
x = np.arange(0, len(y_action))

plt.plot(x, y_joint1, label="joint1")
plt.plot(x, y_action, label="action1")
plt.axhline(0, c="g")
plt.legend()
plt.show()
