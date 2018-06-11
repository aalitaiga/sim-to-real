import numpy as np
import os
import matplotlib.pyplot as plt
import gym
import gym_ergojr

from simple_joints_lstm.lstm_net_real_v3 import LstmNetRealv3

ds = np.load(os.path.expanduser("~/data/sim2real/data-realigned-{}-bullet3.npz".format("train")))

ds_curr_real = ds["ds_curr_real"]
ds_next_real = ds["ds_next_real"]
ds_next_sim = ds["ds_next_sim"]
ds_action = ds["ds_action"]
ds_epi = ds["ds_epi"]

env = gym.make("ErgoFightStatic-Headless-Shield-Move-HalfRand-Bullet-Plus-v0")
_ = env.reset()
old_obs = ds_curr_real[0]
env.set_state(ds_curr_real[0])

for epoch in range(1):
    for idx in range(len(ds_curr_real)):

        print("real t1_x:", np.around(ds_curr_real[idx], 2))
        print("sim_ t1_x:", np.around(old_obs[:12], 2))

        action = ds_action[idx]
        new_obs, _, _, _ = env.step(action)

        print("real t2_x:", np.around(ds_next_real[idx], 2))
        print("sim_ t2_x:", np.around(new_obs[:12], 2))
        old_obs = new_obs

        print("===")
