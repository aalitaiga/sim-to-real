import logging
import numpy as np

from blocks.algorithms import Adam
from blocks.bricks import MLP, Tanh, Identity, BatchNormalizedMLP, LeakyRectifier
from blocks.extensions import FinishAfter, Timing, Printing
from blocks.extensions.monitoring import TrainingDataMonitoring
from blocks.extensions.saveload import Checkpoint
from blocks.filter import VariableFilter
from blocks.graph import (ComputationGraph, apply_batch_normalization,
                          get_batch_normalization_updates, batch_normalization)
from blocks.initialization import IsotropicGaussian, Constant
from blocks.model import Model
from mujoco_py.mjtypes import POINTER, c_double
# from rllab.algos.trpo import TRPO
# from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.gym_env import GymEnv
from rllab.envs.normalized_env import normalize
# from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
# from rllab.misc.instrument import run_experiment_lite
import theano.tensor as T

from main_loop import RLMainLoop

from buffer_ import Buffer, FIFO
from model import GAN

# logging.basicConfig(filename='example.log',
#     level=logging.DEBUG,
#     format='%(message)s')
#
# logging.getLogger().addHandler(logging.StreamHandler())
# logger = logging.getLogger(__name__)

## Definiing the environnement
coeff = 0.85

env = normalize(GymEnv('Swimmer-v1', force_reset=True), normalize_obs=True)
env2 = normalize(GymEnv('Swimmer-v1', force_reset=True), normalize_obs=True)

# The second environnement models the real world
#env2.env.model.opt.gravity = np.array([0, 0, -9.81*coeff]).ctypes.data_as(POINTER(c_double*3)).contents
env2.wrapped_env.env.env.model.opt.gravity = np.array([0, 0, -9.81*coeff]).ctypes.data_as(POINTER(c_double*3)).contents


## Defining the buffer
observation_dim = int(env.observation_space.shape[0])
action_dim = int(env.action_space.shape[0])
rng = np.random.RandomState(seed=23)
max_steps = 10000
history = 2

buffer_ = Buffer(observation_dim, action_dim, rng, history, max_steps)
prev_observations = FIFO(history)
actions = FIFO(history+1)  # also taking current action

## Defining the generative model
# input_gen = history*observation_dim + (history+1)*action_dim
input_dim = (history+1)*(observation_dim + action_dim)
h = 64
LEARNING_RATE = 1e-4
LEARNING_RATE = 1e-4
BETA1 = 0.5
nb_epoch = 30

x = T.matrix('features')
y = T.matrix('transition')

# Doesn't make sense to use batch_norm in online training
# so far the environnement and generative model run synchronously
batch_normalized = False
alpha = 0.05
mlp = BatchNormalizedMLP if batch_normalized else MLP

G = mlp(
    activations=[LeakyRectifier(), LeakyRectifier(), Tanh()],
    dims=[input_dim, h, h, observation_dim],
    name='generator'
)
D = mlp(
    activations=[LeakyRectifier(), LeakyRectifier(), Identity()],
    dims=[input_dim, h, h, 1],
    name='discriminator'
)

generative_model = GAN(G, D, alpha=1., weights_init=IsotropicGaussian(std=0.02, mean=0),
    biases_init=Constant(0.01), name='GAN')
# generative_model.push_allocation_config()
# G.linear_transformations[-1].use_bias = True
generative_model.initialize()
actions_var = env.action_space.new_tensor_variable(
    'actions',
    extra_dims=1
)
obs_prev = env.observation_space.new_tensor_variable(
    'previous_obs',
    extra_dims=1
)
obs_sim = env.observation_space.new_tensor_variable(
    'observation_sim',
    extra_dims=0
)
obs_real = env.observation_space.new_tensor_variable(
    'observation_real',
    extra_dims=0
)
context = T.concatenate([actions_var.flatten(), obs_prev.flatten()])

original_cg = ComputationGraph(generative_model.compute_losses(context, obs_sim, obs_real))

if batch_normalized:
    cg = apply_batch_normalization(original_cg)
    # Add updates for population parameters
    pop_updates = get_batch_normalization_updates(cg)
    extra_updates = [(p, m * alpha + p * (1 - alpha))
                     for p, m in pop_updates]
    import ipdb; ipdb.set_trace()
else:
    cg = original_cg
    extra_updates = []

model = Model(cg.outputs)
discriminator_loss, generator_loss = model.outputs
# import ipdb; ipdb.set_trace()
auxiliary_variables = [u for u in model.auxiliary_variables if 'norm' not in u.name]
auxiliary_variables.extend(VariableFilter(theano_name='squared_error')(model.variables))

step_rule = Adam(learning_rate=LEARNING_RATE, beta1=BETA1)
algorithm = generative_model.algorithm(discriminator_loss, generator_loss, step_rule, step_rule)
#algorithm.add_updates(extra_updates)

d = {
    "env": env,
    "env2": env2,
    "buffer_": buffer_,
    "history": history,
    "render": True,
    "episode_len": 50,
    "trajectory_len": 100,
}

extensions = [
    Timing(),
    FinishAfter(after_n_epochs=1000),
    TrainingDataMonitoring(
        model.outputs+auxiliary_variables,
        # prefix="train",
        after_epoch=True),
    Checkpoint('generative_model.tar', after_n_epochs=10, after_training=True,
               use_cpickle=True),
    Printing()
]

main_loop = RLMainLoop(
    algorithm,
    d,
    model=model,
    extensions=extensions,
)

main_loop.run()

# for i_episode in range(1000):
#     observation = env.reset()
#     observation2 = env2.reset()
#     match_env(env, env2)
#     prev_observations.push(observation)
#
#     for t in range(100):
#         # env.render()
#         # env2.render()
#
#         action = env.action_space.sample()
#         observation, reward, done, info = env.step(action)
#         observation2, reward2, done2, info2 = env2.step(action)
#
#         actions.push(action)
#         if len(prev_observations) == history and len(actions) == history+1:
#             buffer_.add_sample(prev_observations.copy(), actions.copy(), observation, observation2, reward, reward2)
#         prev_observations.push(observation2)
#         match_env(env, env2)
#
#         if done:
#             print("Episode finished after {} timesteps".format(t+1))
#             prev_observations.clear()
#             actions.clear()
#             break
# buffer_.save('/Tmp/alitaiga/sim-to-real/buffer-test')
# buffer_ = Buffer.load('/Tmp/alitaiga/sim-to-real/buffer-test')
#
# f_train = theano.function(
#     inputs=[obs_prev, obs_sim, obs_real, actions_var],
#     outputs=[discriminator_loss],
#     updates=algorithm,
#     allow_input_downcast=True
# )
#
# observations, actions, s_transition, r_transition, _, _ = buffer_.random_batch(1)
# # total_obs = np.vstack([observations[0], s_transition])
# for _ in range(30):
#     result = f_train(observations[0], s_transition[0], r_transition[0], actions[0])
#     print(result)
# import ipdb; ipdb.set_trace()

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
