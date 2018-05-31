import os

from gym_ergojr.sim.single_robot import SingleRobot
import time
import numpy as np
from tqdm import tqdm



for run in ["train", "test"]:
    robot = SingleRobot(debug=True, robot_model="ergojr-penholder")
    ds = np.load(os.path.expanduser("~/data/sim2real/data-realigned-{}.npz".format(run)))

    ds_curr_real = ds["ds_curr_real"]
    ds_next_real = ds["ds_next_real"]
    ds_action = ds["ds_action"]
    ds_epi = ds["ds_epi"]
    ds_next_sim = []

    print(ds_curr_real.shape)

    for i in tqdm(range(len(ds_curr_real))):
        robot.set(ds_curr_real[i])
        # print(robot.observe().round(2))
        robot.act2(ds_action[i])

        robot.step()

        obs = robot.observe()
        ds_next_sim.append(obs)

    ds_next_sim = np.array(ds_next_sim)

    print(ds_next_sim.shape)

    np.savez(os.path.expanduser("~/data/sim2real/data-realigned-{}-bullet3.npz".format(run)),
             ds_curr_real=ds_curr_real,
             ds_next_real=ds_next_real,
             ds_next_sim=ds_next_sim,
             ds_action=ds_action,
             ds_epi=ds_epi
             )
    robot.close()