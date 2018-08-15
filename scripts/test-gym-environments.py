import time

import gym

from gym import envs
for e in envs.registry.all():
    print (e)

#
# env = gym.make("Striker-v2")
#
# env.reset()
# env.render()
#
# for episode in range(10):
#     while True:
#         action = env.action_space.sample()
#         obs, rew, done, misc = env.step(action)
#         env.render()
#         time.sleep(0.1)
#         if done:
#             env.reset()
#             break
#
# env.close()
