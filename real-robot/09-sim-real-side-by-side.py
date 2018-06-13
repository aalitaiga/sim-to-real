import time

import numpy as np
import os
import matplotlib.pyplot as plt
import gym
import gym_ergojr
from gym_ergojr.sim.single_robot import SingleRobot
from s2rr.movements.dataset import DatasetProduction
from itertools import cycle

cycol = cycle('bgrcmk')

ds = DatasetProduction()
ds.load("~/data/sim2real/data-realigned-v3-{}.npz".format("train"))

epi = np.random.randint(0, len(ds.current_real))

joints_sim = np.zeros((299, 6), np.float32)
joints_real_sim = np.zeros((299, 6), np.float32)
joints_real = np.zeros((299, 6), np.float32)

robot_real = SingleRobot(debug=True)
for frame in range(299):
    robot_real.set(ds.current_real[epi, frame])
    # robot_real.act2(ds.current_real[epi, frame, :6])
    robot_real.step()
    joints_real_sim[frame, :] = robot_real.observe()[:6]
    # time.sleep(0.1)
robot_real.close()

robot_sim = SingleRobot(debug=True)
robot_sim.set(ds.current_real[epi, 0])
robot_sim.act2(ds.current_real[epi, 0, :6])
robot_sim.step()
for frame in range(299):
    robot_sim.act2(ds.action[epi, frame])
    robot_sim.step()
    joints_sim[frame, :] = robot_sim.observe()[:6]
    # time.sleep(0.1)
robot_sim.close()

for frame in range(299):
    joints_real[frame, :] = ds.current_real[epi, frame, :6]

for i in range(6):
    c = next(cycol)
    plt.plot(
        np.arange(0, 299),
        joints_real[:, i],
        c="black",
        label="real"

    )
    plt.plot(
        np.arange(0, 299),
        joints_real_sim[:, i],
        c="red",
        dashes=[10, 2],
        label="real_sim"
    )
    plt.plot(
        np.arange(0, 299),
        joints_sim[:, i],
        c="green",
        dashes=[2, 1],
        label="sim"
    )
    plt.plot(
        np.arange(0, 299),
        ds.action[epi, :, i],
        c="blue",
        dashes=[1, 1],
        label="action"
    )
    plt.legend()
    plt.ylim(-1.25, 1.25)

    plt.show()
