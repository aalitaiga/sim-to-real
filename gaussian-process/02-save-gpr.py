from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from simple_joints_lstm.dataset_real_smol_bullet_v2 import DatasetRealSmolBulletV2
import numpy as np
from sklearn.metrics import mean_squared_error


TEST_SAMPLES = 1000

# SAMPLES = 2000
# mse train: 0.00010760481106133677
# mse test: 0.014030914160525777


# SAMPLES = 1000
# mse train: 7.797018350066539e-05
# mse test: 0.014354712721545648


# SAMPLES = 100
# mse train: 5.3951225527470025e-06
# mse test: 0.01523041492122011


# SAMPLES = 10
# mse train: 2.0366736060960247e-18
# mse test: 0.02150001197263307

ds_train = DatasetRealSmolBulletV2(train=True)
ds_test = DatasetRealSmolBulletV2(train=False)

dataset_train_x = []
dataset_train_y = []
dataset_test_x = []
dataset_test_y = []
for i in range(len(ds_train)):
    dataset_train_x.append(ds_train[i]["x"])
    dataset_train_y.append(ds_train[i]["y"])
for i in range(len(ds_test)):
    dataset_test_x.append(ds_test[i]["x"])
    dataset_test_y.append(ds_test[i]["y"])

indices_train = np.random.choice(len(ds_train), SAMPLES)
indices_test = np.random.choice(len(ds_test), TEST_SAMPLES)

dataset_train_x = np.array(dataset_train_x).reshape(len(ds_train)*299,-1)[indices_train]
dataset_train_y = np.array(dataset_train_y).reshape(len(ds_train)*299,-1)[indices_train]
dataset_test_x = np.array(dataset_test_x).reshape(len(ds_test)*299,-1)[indices_test]
dataset_test_y = np.array(dataset_test_y).reshape(len(ds_test)*299,-1)[indices_test]

for ds in [dataset_train_x,dataset_train_y,dataset_test_x,dataset_test_y]:
    print (ds.shape)



kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)

# Fit to data using Maximum Likelihood Estimation of the parameters
gp.fit(dataset_train_x, dataset_train_y)

# Make the prediction on the meshed x-axis (ask for MSE as well)
y_pred, sigma = gp.predict(dataset_train_x, return_std=True)
mse_train = mean_squared_error(y_pred, dataset_train_y)
print("mse train:",mse_train)


# print("sum:", sigma.sum())
# print("mean:", sigma.mean())

y_pred, sigma = gp.predict(dataset_test_x, return_std=True)
mse_test = mean_squared_error(y_pred, dataset_test_y)
print ("mse test:",mse_test)

# print("sum:", sigma.sum())
# print("mean:", sigma.mean())

from sklearn.externals import joblib

joblib.dump(gp, 'models/gp2_{}.pkl'.format(SAMPLES))
