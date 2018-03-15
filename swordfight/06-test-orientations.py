import time
from pypot.creatures import PoppyErgoJr

from poppy_helpers.controller import SwordFightController
from pypot.robot import from_remote


robot_sim = PoppyErgoJr(simulator='vrep')

poppy_att = from_remote('flogo2.local', 4242)

controller_att = SwordFightController(poppy_att, mode="att")
controller_att.rest()


poses = [
    [45,0,0,0,0,0],
    [0,45,0,0,0,0],
    [0,0,45,0,0,0],
    [0,0,0,45,0,0],
    [0,0,0,0,45,0],
    [0,0,0,0,0,45],
]

for pose in poses:
    for i,m in enumerate(robot_sim.motors):
        m.goal_position = pose[i]

    controller_att.goto_pos(pose)

    time.sleep(5)