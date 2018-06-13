import time

import numpy as np
from matplotlib import pyplot as plt
import os
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

DATASET_PATH = os.path.expanduser("~/data/sim2real/data-realigned-{}-bullet3.npz") # {} is either "train" or "test"

MEM = 500

ds_train = np.load(DATASET_PATH.format("train"))

x_train = np.hstack((ds_train["ds_next_sim"], ds_train["ds_curr_real"], ds_train["ds_action"]))
y_train = ds_train["ds_next_real"] - ds_train["ds_curr_real"]

print ("train")
print (x_train.shape)
print (y_train.shape)

ds_test = np.load(DATASET_PATH.format("test"))

x_test = np.hstack((ds_test["ds_next_sim"], ds_test["ds_curr_real"], ds_test["ds_action"]))
y_test = ds_test["ds_next_real"] - ds_test["ds_curr_real"]

print ("test")
print (x_test.shape)
print (y_test.shape)

# Instanciate a Gaussian Process model
kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)


start = time.time()
# Fit to data using Maximum Likelihood Estimation of the parameters
gp.fit(x_train[:MEM], y_train[:MEM])

# Make the prediction on the meshed x-axis (ask for MSE as well)
_, sigma = gp.predict(x_train, return_std=True)

print ("sigma train:",sigma)
#for s in sigma:
#    print(s)

print ("sum train:",sigma.sum())
print ("mean train:",sigma.mean())

_, sigma = gp.predict(x_test, return_std=True)

print ("sigma test:",sigma)

#for s in sigma:
#    print(s)

print ("sum test:",sigma.sum())
print ("mean test:",sigma.mean())





