import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib

matplotlib.style.use('ggplot')

from itertools import cycle

cycol = cycle('bgrcmk')

ds = np.load(os.path.expanduser("~/data/sim2real/data-realigned-{}-bullet3.npz".format("test")))

ds_curr_real = ds["ds_curr_real"]
ds_next_real = ds["ds_next_real"]
ds_next_sim = ds["ds_next_sim"]
ds_action = ds["ds_action"]
ds_epi = ds["ds_epi"]

epi = np.random.randint(0, ds_epi.max())

print(ds_curr_real[ds_epi == epi].shape)

for i in range(6):
    c = next(cycol)
    plt.plot(
        np.arange(0, len(ds_next_real[ds_epi == epi])),
        ds_next_real[ds_epi == epi][:, i],
        c=c
    )
    plt.plot(
        np.arange(0, len(ds_next_real[ds_epi == epi])),
        ds_next_sim[ds_epi == epi][:, i],
        c=c,
        dashes=[10,2]
    )

plt.show()
