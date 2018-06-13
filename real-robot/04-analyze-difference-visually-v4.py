import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib
from s2rr.movements.dataset import DatasetProduction

matplotlib.style.use('ggplot')

from itertools import cycle

cycol = cycle('bgrcmk')

ds = DatasetProduction()
ds.load("~/data/sim2real/data-realigned-v3-{}-bullet.npz".format("test"))


epi = np.random.randint(0, len(ds.current_real))

print(ds.current_real[epi].shape)

for i in range(6):
    c = next(cycol)
    plt.plot(
        np.arange(0, len(ds.next_real[epi])),
        ds.next_real[epi,:,i],
        c=c
    )
    plt.plot(
        np.arange(0, len(ds.next_real[epi])),
        ds.next_sim[epi, :, i],
        c=c,
        dashes=[10,2]
    )

plt.show()
