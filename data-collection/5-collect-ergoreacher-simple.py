import os

import gym
import gym_ergojr
import numpy as np

# real = backlash
# train/eval

# real/real: 0.5292118170308328
# sim/sim: 0.5513551936027279
# sim/real: 0.4971467640693259
# real/sim: 0.5186317888036542


# real/real 1: 0.507
# real/real 2: 0.527
#
# real/real 1: 0.3558040105157774
# real/real 2: 0.3137
# real/sim: 1: 0.451
# real/sim: 2: 0.1176
# sim/real 1: -0.418
# sim/real 2: -5.568
# sim/real 2: -10.234
from tqdm import tqdm

MAX_EPISODES = 1000
EPISODE_LEN = 100
ACTION_STEPS = 1  # must be >= 1
OUTPUT_FILE = "~/data/sim2real/data-ergoreachersimple-v1.npz"

env_real = gym.make("ErgoReacher-Headless-Simple-Backlash-v1")
env_sim = gym.make("ErgoReacher-Headless-Simple-v1")

data_current_real = np.zeros((MAX_EPISODES, EPISODE_LEN, 8), dtype=np.float32)
data_next_real = np.zeros((MAX_EPISODES, EPISODE_LEN, 8), dtype=np.float32)
data_next_sim = np.zeros((MAX_EPISODES, EPISODE_LEN, 8), dtype=np.float32)
data_action = np.zeros((MAX_EPISODES, EPISODE_LEN, 4), dtype=np.float32)


def dump(state_current_real, state_next_real, state_next_sim, actions):
    np.savez(os.path.expanduser(OUTPUT_FILE),
             state_current_real=state_current_real,
             state_next_real=state_next_real,
             state_next_sim=state_next_sim,
             actions=actions)



for episode in tqdm(range(MAX_EPISODES)):
    obs_real_old = env_real.reset()
    env_sim.reset()
    env_sim.unwrapped._set_state(env_real.unwrapped._get_state())

    action = np.random.uniform(-1, +1, 4)
    action_step = 0
    for frame in range(EPISODE_LEN):
        action_step += 1
        if action_step == ACTION_STEPS:
            action = np.random.uniform(-1, +1, 4)
            action_step = 0

        obs_real, _, _, _ = env_real.step(action)
        obs_sim, _, _, _ = env_sim.step(action)

        data_action[episode, frame, :] = action.copy()
        data_current_real[episode, frame, :] = obs_real_old[:8].copy()
        data_next_real[episode, frame, :] = obs_real[:8].copy()
        data_next_sim[episode, frame, :] = obs_sim[:8].copy()

        env_sim.unwrapped._set_state(env_real.unwrapped._get_state())

        obs_real_old = obs_real.copy()

    dump(
        data_current_real,
        data_next_real,
        data_next_sim,
        data_action
    )

print("saving done to:", OUTPUT_FILE)