import numpy as np
import matplotlib.pyplot as plt
import torch
import gym
import gym_ergojr


# good results for jittery at seed = 0
np.random.seed(1)


def make_actions(stepping):
    actions = np.zeros((100, 4), dtype=np.float32)
    for i in range(100):
        if i % stepping == 0:
            action = np.random.uniform(low=-1, high=1, size=4)
        actions[i] = action
    return actions


actions_smooth = make_actions(20)
actions_jitter = make_actions(1)

for act in [actions_smooth, actions_jitter]:
    for i in range(21):
        print(i, act[i])

traj_smoo_sim = np.zeros((100, 4), dtype=np.float32)
traj_smoo_real = np.zeros((100, 4), dtype=np.float32)
traj_smoo_simplus = np.zeros((100, 4), dtype=np.float32)

traj_jitt_sim = np.zeros((100, 4), dtype=np.float32)
traj_jitt_real = np.zeros((100, 4), dtype=np.float32)
traj_jitt_simplus = np.zeros((100, 4), dtype=np.float32)

# record trajectories smooth

# load bullet env sim
env = gym.make("ErgoReacher-Headless-Simple-v1")
env.reset()
env.unwrapped._set_state([0] * 8)
for i in range(100):
    obs, _, _, _ = env.step(actions_smooth[i])
    traj_smoo_sim[i] = obs[:4].copy()
env.close()

# load bullet env real
env = gym.make("ErgoReacher-Headless-Simple-Backlash-v1")
env.reset()
env.unwrapped._set_state([0] * 8)
for i in range(100):
    obs, _, _, _ = env.step(actions_smooth[i])
    traj_smoo_real[i] = obs[:4].copy()
env.close()

# load bullet env sim+
env = gym.make("ErgoReacher-Headless-Simple-Plus-v2")
env.reset()
env.unwrapped._set_state([0] * 8)
for i in range(100):
    obs, _, _, _ = env.step(actions_smooth[i])
    traj_smoo_simplus[i] = obs[:4].copy()
env.close()

# record trajectories jittery

# load bullet env sim
env = gym.make("ErgoReacher-Headless-Simple-v1")
env.reset()
env.unwrapped._set_state([0] * 8)
for i in range(100):
    obs, _, _, _ = env.step(actions_jitter[i])
    traj_jitt_sim[i] = obs[:4].copy()
env.close()

# load bullet env real
env = gym.make("ErgoReacher-Headless-Simple-Backlash-v1")
env.reset()
env.unwrapped._set_state([0] * 8)
for i in range(100):
    obs, _, _, _ = env.step(actions_jitter[i])
    traj_jitt_real[i] = obs[:4].copy()
env.close()

# load bullet env sim+
env = gym.make("ErgoReacher-Headless-Simple-Plus-v2")
env.reset()
env.unwrapped._set_state([0] * 8)
for i in range(100):
    obs, _, _, _ = env.step(actions_jitter[i])
    traj_jitt_simplus[i] = obs[:4].copy()
env.close()

# plot


x = np.arange(100)



for trajs, acts in [
    ([traj_jitt_real, traj_jitt_sim, traj_jitt_simplus], actions_jitter),
    ([traj_smoo_real, traj_smoo_sim, traj_smoo_simplus], actions_smooth)
]:
    fig, ax = plt.subplots(2, 2, sharex='col', sharey='row')
    ax[0, 0].scatter(x, acts[:, 0], label="actions")
    ax[0, 0].plot(x, trajs[0][:, 0], label="real")
    ax[0, 0].plot(x, trajs[1][:, 0], label="sim")
    ax[0, 0].plot(x, trajs[2][:, 0], label="simplus")

    ax[0, 1].scatter(x, acts[:, 1], label="actions")
    ax[0, 1].plot(x, trajs[0][:, 1], label="real")
    ax[0, 1].plot(x, trajs[1][:, 1], label="sim")
    ax[0, 1].plot(x, trajs[2][:, 1], label="simplus")

    ax[1, 0].scatter(x, acts[:, 2], label="actions")
    ax[1, 0].plot(x, trajs[0][:, 2], label="real")
    ax[1, 0].plot(x, trajs[1][:, 2], label="sim")
    ax[1, 0].plot(x, trajs[2][:, 2], label="simplus")

    ax[1, 1].scatter(x, acts[:, 3], label="actions")
    ax[1, 1].plot(x, trajs[0][:, 3], label="real")
    ax[1, 1].plot(x, trajs[1][:, 3], label="sim")
    ax[1, 1].plot(x, trajs[2][:, 3], label="simplus")

    plt.legend()
    plt.show()
