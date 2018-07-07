import os

from gym_ergojr.sim.single_robot import SingleRobot
import time
import numpy as np
from s2rr.movements.dataset import DatasetProduction
from tqdm import tqdm

for run in ["train", "test"]:
    robot = SingleRobot(debug=False, robot_model="ergojr-penholder")
    ds = DatasetProduction()
    ds.load("~/data/sim2real/data-realigned-v3-{}.npz".format(run))

    print(ds.current_real.shape)

    for epi in tqdm(range(len(ds.current_real))):
        for frame in range(len(ds.current_real[0])):
            robot.set(ds.current_real[epi, frame, :])
            robot.step()
            # print(robot.observe().round(2))
            robot.act2(ds.action[epi, frame, :])

            robot.step()

            obs = robot.observe()
            ds.next_sim[epi, frame, :] = obs

    ds.save("~/data/sim2real/data-realigned-v4-{}-bullet.npz".format(run))

    print("ds.next_sim.max()",ds.next_sim.max())
    print("ds.next_sim.min()",ds.next_sim.min())

    robot.close()
