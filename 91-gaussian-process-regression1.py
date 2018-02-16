import numpy as np
from matplotlib import pyplot as plt

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

DATASET_PATH = "/home/florian/data/sim2real/dataset-real-{}.npz" # {} is either "train" or "test"

MEM = 10000

ds_train = np.load(DATASET_PATH.format("train"))

x_train = ds_train["ds_in"]
y_train = ds_train["ds_diff"]

ds_test = np.load(DATASET_PATH.format("test"))

x_test = ds_train["ds_in"]
y_test = ds_train["ds_diff"]

# Instanciate a Gaussian Process model
kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)



# Fit to data using Maximum Likelihood Estimation of the parameters
gp.fit(x_train[:MEM], y_train[:MEM])

# Make the prediction on the meshed x-axis (ask for MSE as well)
_, sigma = gp.predict(x_train, return_std=True)

print ("sigma train:",sigma)

_, sigma = gp.predict(x_test, return_std=True)

print ("sigma train:",sigma)




