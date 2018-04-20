from poppy_helpers.controller import SwordFightController
from pypot.robot import from_remote

poppy_def = from_remote('flogo4.local', 4242)
poppy_att = from_remote('flogo2.local', 4242)

controller_def = SwordFightController(poppy_def, mode="def")
controller_att = SwordFightController(poppy_att, mode="att")

controller_def.compliant(False)
controller_att.compliant(False)

controller_att.set_max_speed(100)
controller_def.set_max_speed(100)

controller_def.safe_rest()
controller_att.safe_rest()

#dddd
