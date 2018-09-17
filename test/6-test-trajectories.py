import numpy as np
import matplotlib.pyplot as plt
import torch
import gym
import gym_ergojr

# load bullet env sim
env_sim = gym.make("ErgoReacher-Headless-Simple-v1")

# load bullet env real
env_real = gym.make("ErgoReacher-Headless-Simple-Backlash-v1")

# load bullet env sim+
env_simplus = gym.make("ErgoReacher-Headless-Simple-Plus-v1")

# reset
for env in [env_sim, env_real, env_real]:
    env.reset()

# record trajectories jittery


# plot

# record trajectories smooth

# plot






















