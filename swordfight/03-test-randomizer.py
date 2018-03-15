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

rand_def = rand.random_sf()
rand_att = rand.random_sf()

print (rand_def, rand_att)

controller_att.set_speed(50)
controller_def.set_speed(50)

controller_def.goto_pos(rand_def)
controller_att.goto_pos(rand_att)

for _ in range(20):
    pos_left = controller_def.get_pos()
    pos_right = controller_att.get_pos()

    print (pos_left, pos_right)

controller_def.rest()
controller_att.rest()


