import time
from pypot.robot import from_remote
import numpy as np
from gym_ergojr.sim.single_robot import SingleRobot

robot_sim = SingleRobot(debug=True)
robot_real = from_remote('flogo2.local', 4242)


poses = [
    [45,0,0,0,0,0],
    [0,45,0,0,0,0],
    [0,0,45,0,0,0],
    [0,0,0,45,0,0],
    [0,0,0,0,45,0],
    [0,0,0,0,0,45],
]

for pose in poses:
    pose_sim = np.deg2rad(pose+[0]*6)/(np.pi/2)
    robot_sim.set(pose_sim)
    robot_sim.act2(pose_sim[:6])
    # robot_sim.step()
    for i in range(100):
        robot_sim.step()

    print(robot_sim.observe().round(1))
    for i,m in enumerate(robot_real.motors):
        m.goal_position = pose[i]

    time.sleep(2)