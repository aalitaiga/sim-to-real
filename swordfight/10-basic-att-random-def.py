import time
from poppy_helpers.normalizer import Normalizer
from pypot.robot import from_remote
from pytorch_a2c_ppo_acktr.inference import Inference

from poppy_helpers.controller import SwordFightController
from poppy_helpers.randomizer import Randomizer

import numpy as np

from poppy_helpers.startups import startup_swordfight

controller_att, controller_def = startup_swordfight("flogo2","flogo4")

# random defense
rand = Randomizer()
# rand_def = rand.random_sf()
# controller_def.goto_pos(rand_def)

norm = Normalizer()

inf = Inference("/home/florian/dev/pytorch-a2c-ppo-acktr/"
                "trained_models/ppo/"
                "ErgoFightStatic-Headless-Fencing-v0-180209225957.pt")

# ErgoFightStatic-Headless-Fencing-v0-180301140951.pt - very stabby, bit dumb
# ErgoFightStatic-Headless-Fencing-v0-180301140937.pt - pretty good, slightly stabby
# ErgoFightStatic-Headless-Fencing-v0-180301140520.pt - ultimate stabby policy

# ErgoFightStatic-Headless-Fencing-v0-180209225957.pt - aggressive lunge policy, good tracking, best policy


def get_observation():
    norm_pos_att = norm.normalize_pos(controller_att.get_pos_comp())
    norm_vel_att = norm.normalize_vel(controller_att.get_current_speed())
    norm_pos_def = norm.normalize_pos(controller_def.get_pos_comp())
    norm_vel_def = norm.normalize_vel(controller_def.get_current_speed())

    return np.hstack((norm_pos_att, norm_vel_att, norm_pos_def, norm_vel_def))

print ("=== starting main loop")
while True:

# start = time.time()
    for i in range(20):
        if i % 10 == 0:
            controller_def.goto_pos(rand.random_def_stance())
        action = inf.get_action(get_observation())
        action = np.clip(action, -1, 1)
        action = norm.denormalize_pos(action)
        controller_att.act(action, scaling=0.2)
    # diff = time.time()-start
    # print ("fps:",100/diff)

    controller_att.safe_rest()
    controller_def.safe_rest()

    time.sleep(2)



