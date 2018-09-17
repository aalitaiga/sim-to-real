import os

import gym
import gym_ergojr
import numpy as np

from tqdm import tqdm

MAX_EPISODES = 1000
EPISODE_LEN = 100
ACTION_STEPS = 1  # must be >= 1
INOUTPUT_FILE = "~/data/sim2real/data-ergoreachersimple-v2.npz"

env_sim = gym.make("ErgoReacher-Headless-Simple-v1")

data = np.load(os.path.expanduser(INOUTPUT_FILE))

data_current_real = data["state_current_real"]
data_next_real = data["state_next_real"]
data_next_sim = data["state_next_real"]
data_action = data["actions"]


def dump(state_current_real, state_next_real, state_next_sim, actions):
    np.savez(os.path.expanduser(INOUTPUT_FILE),
             state_current_real=state_current_real,
             state_next_real=state_next_real,
             state_next_sim=state_next_sim,
             actions=actions)



for episode in tqdm(range(MAX_EPISODES)):
    obs_real_old = data_current_real[episode,0,:]
    env_sim.reset()
    env_sim.unwrapped._set_state(obs_real_old)

    for frame in range(EPISODE_LEN):
        obs_sim, _, _, _ = env_sim.step(data_action[episode, frame, :])
        data_next_sim[episode, frame, :] = obs_sim[:8].copy()
        env_sim.unwrapped._set_state(data_next_real[episode, frame, :])

    dump(
        data_current_real,
        data_next_real,
        data_next_sim,
        data_action
    )

print("saving done to:", INOUTPUT_FILE)