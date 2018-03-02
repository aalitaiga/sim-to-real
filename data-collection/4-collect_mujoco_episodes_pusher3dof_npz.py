import math

import gym
import gym_throwandpush
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# from hyperdash import Experiment


max_steps = 1000
episode_length = 100  # how many steps max in each rollout?
# split = 0.90 # train/test split is done in the dataloader by reserving the first 100 elements for validation
action_steps = 1
OUTPUT_FILE = "/tmp/mujoco_pusher3dof_simple_{}act.npz".format(action_steps)

env_sim = gym.make('Pusher3Dof2-v0')  # sim
env_real = gym.make('Pusher3Dof2-v0')  # real

env_sim.env._init(  # sim
    torques=[1, 1, 1],
    colored=True
)
env_sim.reset()

env_real.env._init(  # real
    torques=[1, 1, 1],
    xml='3link_gripper_push_2d_backlash',
    colored=False
)
env_real.reset()

# size_train = math.floor(max_steps * split)

state_current_real = np.zeros((max_steps, episode_length, 12), dtype=np.float32)
state_next_sim = np.zeros((max_steps, episode_length, 12), dtype=np.float32)
state_next_real = np.zeros((max_steps, episode_length, 12), dtype=np.float32)
actions = np.zeros((max_steps, episode_length, 3), dtype=np.float32)


def match_env(ev1, ev2):
    # set env1 (simulator) to that of env2 (real robot)
    ev1.env.set_state(
        ev2.env.model.data.qpos.ravel(),
        ev2.env.model.data.qvel.ravel()
    )


i = 0


def dump(state_current_real, state_next_real, state_next_sim, actions):
    np.savez(OUTPUT_FILE,
             state_current_real=state_current_real,
             state_next_real=state_next_real,
             state_next_sim=state_next_sim,
             actions=actions)
    print("saving done to:", OUTPUT_FILE)


for i in tqdm(range(max_steps)):
    obs_sim = env_sim.reset()
    obs_real = env_real.reset()
    match_env(env_sim, env_real)

    for j in range(episode_length):

        if j % action_steps == 0:
            action = env_sim.action_space.sample()

        obs_sim_next, reward_sim, _, _ = env_sim.step(action.copy())
        obs_real_next, reward_real, done, _ = env_real.step(action.copy())

        state_current_real[i, j, :] = obs_real
        state_next_real[i, j, :] = obs_real_next
        state_next_sim[i, j, :] = obs_sim_next
        actions[i, j, :] = action

        obs_real = obs_real_next

        match_env(env_sim, env_real)

        if done:
            break

    if i % 100 == 0:
        print("{} done".format(i))
        dump(state_current_real, state_next_real, state_next_sim, actions)