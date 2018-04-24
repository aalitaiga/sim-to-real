import time
from pypot.robot import from_remote

from poppy_helpers.constants import SWORDFIGHT_REST_DEF, SWORDFIGHT_REST_ATT
from poppy_helpers.controller import Controller, SwordFightController
from poppy_helpers.randomizer import Randomizer

poppy_def = from_remote('flogo4.local', 4242)
poppy_att = from_remote('flogo2.local', 4242)

controller_def = SwordFightController(poppy_def, mode="def")
controller_att = SwordFightController(poppy_att, mode="att")

rand = Randomizer()

controller_def.rest()
controller_att.rest()

controller_att.set_max_speed(50)
controller_def.set_max_speed(50)

speeds = []

for i in range(100):
    if i % 10 == 0:
        rand_def = rand.random_sf()
        rand_att = rand.random_sf()
        controller_def.goto_pos(rand_def)
        controller_att.goto_pos(rand_att)

    vel_left = controller_def.get_current_speed()
    vel_right = controller_att.get_current_speed()

    for vel in vel_left+vel_right:
        speeds.append(vel)

controller_def.rest()
controller_att.rest()

print (max(speeds), min(speeds), sum(speeds)/len(speeds))
