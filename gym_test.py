import gym
# import numpy as np
# from mujoco_py.mjtypes import POINTER, c_double
from rllab.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.gym_env import GymEnv
from rllab.envs.normalized_env import normalize
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.misc.instrument import run_experiment_lite

# coeff = 0.5

env = normalize(GymEnv('Swimmer-v1', force_reset=True))
#env = gym.make('HalfCheetah-v1')
# env2 = gym.make('HalfCheetah-v1')
#
# env2.env.model.opt.gravity = np.array([0, 0, -9.81*coeff]).ctypes.data_as(POINTER(c_double*3)).contents
#
# def match_env(env1, env2):
#     # make env2 state match env1 state
#     env2.env.set_state(env1.env.model.data.qpos.ravel(), env1.env.model.data.qvel.ravel())

# for i_episode in range(20):
#     observation = env.reset()
#     observation2 = env2.reset()
#     match_env(env, env2)
#     for t in range(100):
#         env.render()
#         env2.render()
#         print(observation)
#         print(observation2)
#         action = env.action_space.sample()
#         #action2 = env2.action_space.sample()
#         observation, reward, done, info = env.step(action)
#         observation2, reward2, done2, info2 = env2.step(action)
#         if done:
#             print("Episode finished after {} timesteps".format(t+1))
#             break


def run_task(*_):
    policy = GaussianMLPPolicy(
        env_spec=env.spec,
        # The neural network policy should have two hidden layers, each with 32 hidden units.
        hidden_sizes=(50, 50)
    )

    baseline = LinearFeatureBaseline(env_spec=env.spec)

    algo = TRPO(
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=4000,
        whole_paths=True,
        max_path_length=200,
        n_itr=40,
        discount=0.99,
        step_size=0.01,
    )
    algo.train()

run_experiment_lite(
    run_task,
    # Number of parallel workers for sampling
    n_parallel=1,
    # Only keep the snapshot parameters for the last iteration
    snapshot_mode="last",
    # Specifies the seed for the experiment. If this is not provided, a random seed
    # will be used
    seed=1,
    #plot=True,
)
