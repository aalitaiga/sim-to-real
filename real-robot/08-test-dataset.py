import numpy as np
import os
import matplotlib.pyplot as plt
import gym
import gym_ergojr
from s2rr.movements.dataset import DatasetProduction
from simple_joints_lstm.lstm_net_real_v3 import LstmNetRealv3



ds = DatasetProduction()
ds.load("~/data/sim2real/data-realigned-v3-{}-bullet.npz".format("train"))

#### POS
# print("ds.next_real.max()",ds.next_real[:,:,:6].max())
# print("ds.next_real.min()",ds.next_real[:,:,:6].min())
#
# print("ds.next_sim.max()",ds.next_sim[:,:,:6].max())
# print("ds.next_sim.min()",ds.next_sim[:,:,:6].min())
#
# print("ds.next_sim.argmax()",ds.next_sim[:,:,:6].argmax())
# print("ds.next_sim.argmin()",ds.next_sim[:,:,:6].argmin())


#### VEL
print("ds.next_real.max()",ds.next_real[:,:,6:].max())
print("ds.next_real.min()",ds.next_real[:,:,6:].min())

print("ds.next_sim.max()",ds.next_sim[:,:,6:].max())
print("ds.next_sim.min()",ds.next_sim[:,:,6:].min())

print("ds.next_sim.argmax()",ds.next_sim[:,:,6:].argmax())
print("ds.next_sim.argmin()",ds.next_sim[:,:,6:].argmin())


n, bins, patches = plt.hist(ds.next_sim[:,:,6:].flatten(), 50, normed=1, facecolor='green', alpha=0.75)
n, bins, patches = plt.hist(ds.next_sim[:,:,:6].flatten(), 50, normed=1, facecolor='red', alpha=0.75)

plt.show()

n, bins, patches = plt.hist(ds.next_real[:,:,6:].flatten(), 50, normed=1, facecolor='green', alpha=0.75)
n, bins, patches = plt.hist(ds.next_real[:,:,:6].flatten(), 50, normed=1, facecolor='red', alpha=0.75)

plt.show()