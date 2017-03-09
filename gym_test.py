import gym
import numpy as np
from mujoco_py.mjtypes import POINTER, c_double

coeff = 1

env = gym.make('HalfCheetah-v1')
env2 = gym.make('HalfCheetah-v1')

env.env.model.opt.gravity = np.array([0, 0, -9.81*coeff]).ctypes.data_as(POINTER(c_double*3)).contents

def match_env(env1, env2):
    # make env2 state match env1 state
    env2.env.set_state(env1.env.model.data.qpos.ravel(), env1.env.model.data.qvel.ravel())

for i_episode in range(20):
    observation = env.reset()
    observation2 = env2.reset()
    match_env(env, env2)
    for t in range(100):
        env.render()
        env2.render()
        print(observation)
        print(observation2)
        action = env.action_space.sample()
        #action2 = env2.action_space.sample()
        observation, reward, done, info = env.step(action)
        observation2, reward2, done2, info2 = env2.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
