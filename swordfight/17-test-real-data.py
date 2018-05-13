import numpy as np
import matplotlib.pyplot as plt
import os

DATASET_PATH = "~/data/sim2real/dataset-real-{}-normalized.npz" # {} is either "train" or "test"



ds_test = np.load(os.path.expanduser(DATASET_PATH.format("test")))

x_test = ds_test["ds_in"]
y_test = ds_test["ds_diff"]

print (x_test.shape)
print (y_test.shape)


ds_train = np.load(os.path.expanduser(DATASET_PATH.format("train")))

x_train = ds_train["ds_in"]
y_train = ds_train["ds_diff"]

print (x_train.shape)
print (y_train.shape)

print (x_train.max())
print (x_train.min())
print (y_train.max())
print (y_train.min())

positions_flat = x_train[:,12:18].flatten()
velocities_flat = x_train[:,18:24].flatten()
actions_flat = x_train[:,24:].flatten()

# plt.hist(positions_flat, 100)
# plt.hist(velocities_flat)
plt.hist(actions_flat)
plt.show()
