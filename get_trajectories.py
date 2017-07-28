#!/usr/bin/env python

from mujoco_py.mjtypes import POINTER, c_double
import numpy as np
# from rllab.algos.trpo import TRPO
# from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.gym_env import GymEnv
from rllab.envs.normalized_env import normalize
# from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
# from rllab.misc.instrument import run_experiment_lite

from utils.buffer_ import Buffer, FIFO

## Definiing the environnement
coeff = 0.85
noise_std = 0.8

env = normalize(GymEnv('Swimmer-v1', force_reset=True), normalize_obs=True)
env2 = normalize(GymEnv('Swimmer-v1', force_reset=True), normalize_obs=True)

def match_env(ev1, ev2):
    # make env1 state match env2 state (simulator matches real world)
    ev1.wrapped_env.env.env.set_state(
        ev2.wrapped_env.env.env.model.data.qpos.ravel(),
        ev2.wrapped_env.env.env.model.data.qvel.ravel()
    )

# The second environnement models the real world
#env2.env.model.opt.gravity = np.array([0, 0, -9.81*coeff]).ctypes.data_as(POINTER(c_double*3)).contents
# env2.wrapped_env.env.env.model.opt.gravity = np.array([0, 0, -9.81*coeff]).ctypes.data_as(POINTER(c_double*3)).contents


## Defining the buffer
observation_dim = int(env.observation_space.shape[0])
action_dim = int(env.action_space.shape[0])
rng = np.random.RandomState(seed=23)
max_steps = 1000000
history = 2

buffer_ = Buffer(observation_dim, action_dim, rng, history, max_steps)
prev_observations = FIFO(history)
actions = FIFO(history)

# for i_episode in range(1000):
while not buffer_.full:
    observation = env.reset()
    observation2 = env2.reset()
    # match_env(env, env2)
    prev_observations.push(observation)

    for t in range(100):
        env.render()
        env2.render()

        action = env.action_space.sample()
        # add correlated/uncorrelated noise
        noisy_action = action + np.random.normal(scale=noise_std, size=action_dim)
        observation, reward, done, info = env.step(action)
        observation2, reward2, done2, info2 = env2.step(noisy_action)

        actions.push(action)
        if len(prev_observations) == history and len(actions) == history:
            buffer_.add_sample(prev_observations.copy(), actions.copy(), observation, observation2, reward, reward2)
        prev_observations.push(observation2)
        # match_env(env, env2)

        if done:
            print("Episode finished after {} timesteps".format(t+1))
            prev_observations.clear()
            actions.clear()
            break

buffer_.save('/Tmp/alitaiga/sim-to-real/buffer_simmer_h2_random')

# buffer_ = Buffer.load('/Tmp/alitaiga/sim-to-real/buffer-test')

# def run_task(*_):
#     policy = GaussianMLPPolicy(
#         env_spec=env.spec,
#         # The neural network policy should have two hidden layers, each with 32 hidden units.
#         hidden_sizes=(50, 50)
#     )
#
#     baseline = LinearFeatureBaseline(env_spec=env.spec)
#
#     algo = TRPO(
#         env=env,
#         policy=policy,
#         baseline=baseline,
#         batch_size=4000,
#         whole_paths=True,
#         max_path_length=100,
#         n_itr=40,
#         discount=0.99,
#         step_size=0.01,
#     )
#     algo.train()
#
# run_experiment_lite(
#     run_task,
#     # Number of parallel workers for sampling
#     n_parallel=1,
#     # Only keep the snapshot parameters for the last iteration
#     snapshot_mode="last",
#     # Specifies the seed for the experiment. If this is not provided, a random seed
#     # will be used
#     seed=1,
#     #plot=True,
# )
