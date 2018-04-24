from pypot.robot import from_remote

from poppy_helpers.constants import SWORDFIGHT_REST_DEF, SWORDFIGHT_REST_ATT
from poppy_helpers.controller import Controller, SwordFightController

poppy_def = from_remote('flogo4.local', 4242)
poppy_att = from_remote('flogo2.local', 4242)

controller_def = SwordFightController(poppy_def, mode="def")
controller_att = SwordFightController(poppy_att, mode="att")

controller_def.rest()
controller_att.rest()
