from poppy_helpers.controller import SwordFightController, SwordFightZMQController

controller_def = SwordFightZMQController(mode="def", host="flogo4.local")
controller_att = SwordFightZMQController(mode="att", host="flogo2.local")

controller_def.compliant(False)
controller_att.compliant(False)

controller_att.set_max_speed(100)
controller_def.set_max_speed(100)

controller_def.safe_rest()
controller_att.safe_rest()


#dddd
#
#
#
#