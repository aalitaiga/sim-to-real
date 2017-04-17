#!/usr/bin/env python

import logging
import numpy as np

from blocks.algorithms import Adam, RMSProp, GradientDescent
from blocks.bricks import MLP, Tanh, Identity, BatchNormalizedMLP, LeakyRectifier
from blocks.extensions import FinishAfter, Printing, ProgressBar
from blocks.extensions.monitoring import DataStreamMonitoring, TrainingDataMonitoring
from blocks.extensions.saveload import Checkpoint
from blocks.graph import (ComputationGraph, apply_batch_normalization,
                          get_batch_normalization_updates)
from blocks.initialization import IsotropicGaussian, Constant
from blocks.model import Model
from fuel.streams import DataStream
from fuel.schemes import ShuffledScheme
from fuel.datasets.hdf5 import H5PYDataset
from rllab.envs.gym_env import GymEnv
from rllab.envs.normalized_env import normalize
import theano.tensor as T

from buffer_ import Buffer, buffer_to_h5
# from generative_model import WGAN, WeightClipping
from original_main_loop import MainLoop

# logging.basicConfig(filename='example.log',
#     level=logging.DEBUG,
#     format='%(message)s')
#
# logging.getLogger().addHandler(logging.StreamHandler())
# logger = logging.getLogger(__name__)

# env = normalize(GymEnv('Swimmer-v1', force_reset=True), normalize_obs=True)
history = 3

## Defining the buffer
buffer_ = Buffer.load('/Tmp/alitaiga/sim-to-real/buffer_swimmer_h{}_random'.format(history))
correction_dataset = buffer_.r_transition - buffer_.s_transition
mean = correction_dataset.mean()
var = correction_dataset.var()

observation_dim = buffer_.observation_dim
action_dim = buffer_.action_dim
rng = np.random.RandomState(seed=23)
del buffer_
del correction_dataset

## Defining the predictive model
input_dim = (history+1)*observation_dim + history*action_dim
h = 512
LEARNING_RATE = 5e-3
BETA1 = 0.5
BATCH = 32
batch_norm = True
alpha = 0.1
mlp = BatchNormalizedMLP if batch_norm else MLP
extra_kwargs = {'conserve_memory': False} if batch_norm else {}

# Doesn't make sense to use batch_norm in online training
# so far the environnement and generative model run synchronously
# actions_var = env.action_space.new_tensor_variable(
#     'actions',
#     extra_dims=2
# )
# obs_prev = env.observation_space.new_tensor_variable(
#     'previous_obs',
#     extra_dims=2
# )
# obs_sim = env.observation_space.new_tensor_variable(
#     'observation_sim',
#     extra_dims=1
# )
# obs_real = env.observation_space.new_tensor_variable(
#     'observation_real',
#     extra_dims=1
# )
actions_var = T.tensor3('actions', dtype='float32')
obs_prev = T.tensor3('previous_obs', dtype='float32')
obs_sim = T.matrix('observation_sim', dtype='float32')
obs_real = T.matrix('observation_real', dtype='float32')

# context = T.concatenate([actions_var.reshape([BATCH, -1]), obs_prev.reshape([BATCH, -1])], axis=1)
context = T.concatenate(
    [actions_var.reshape([BATCH, -1]), obs_prev.reshape([BATCH, -1]), obs_sim.reshape([BATCH, -1])],
    axis=1
)

# G = MLP(
#     activations=[LeakyRectifier(), LeakyRectifier(), Tanh()],
#     dims=[input_dim, h, h, observation_dim],
#     name='generator'
# )
# D = MLP(
#     activations=[LeakyRectifier(), LeakyRectifier(), Identity()],
#     dims=[input_dim, h, h, 1],
#     name='discriminator'
# )

# generative_model = WGAN(G, D, alpha=0, weights_init=IsotropicGaussian(std=0.01, mean=0),
#     biases_init=Constant(0.01), name='WGAN')
# generative_model.initialize()
#
# model = Model(generative_model.losses(context, obs_sim, obs_real))
# discriminator_loss, generator_loss = model.outputs
#
# step_rule = RMSProp(learning_rate=LEARNING_RATE)
# discriminator_algo, generator_algo = generative_model.algorithm(discriminator_loss, generator_loss, step_rule, step_rule)

predictive_model = mlp(
    activations=[LeakyRectifier(), LeakyRectifier(), Identity()],
    dims=[input_dim, h, h, observation_dim],
    weights_init=IsotropicGaussian(std=0.01, mean=0),
    biases_init=Constant(0.01),
    name='predictive_model',
    **extra_kwargs
)
predictive_model.initialize()
correction_predicted = predictive_model.apply(context)
correction = obs_real - obs_sim
correction_scaled = (correction - mean) / np.sqrt(var)
loss = T.sqr(correction_scaled - correction_predicted).mean()
loss.name = 'squared_error'

obs_predicted = (correction_predicted * np.sqrt(var)) + mean + obs_sim
percent = 100*abs((obs_predicted - obs_real) / obs_real).mean()
percent.name = "percent_correction_predicted"

percent_sim = 100*abs(correction / obs_real).mean()
percent_sim.name = "percent_correction_sim"

original_cg = ComputationGraph(loss)

if batch_norm:
    cg = apply_batch_normalization(original_cg)
    # Add updates for population parameters
    pop_updates = get_batch_normalization_updates(cg)
    extra_updates = [(p, m * alpha + p * (1 - alpha))
                         for p, m in pop_updates]
else:
    cg = original_cg
    extra_updates = []

model = Model(cg.outputs)
algorithm = GradientDescent(
    cost=loss,
    parameters=model.parameters,
    step_rule=Adam(learning_rate=LEARNING_RATE, beta1=BETA1),
    on_unused_sources='ignore',
    theano_func_kwargs={'allow_input_downcast': True}
)

# auxiliary_variables = [u for u in model.auxiliary_variables if 'percent_error' in u.name]
swimmer_train = H5PYDataset('/Tmp/alitaiga/sim-to-real/swimmer_h{}_random.h5'.format(history), which_sets=('train',),
        sources=('previous_obs','actions','observation_sim', 'observation_real'))
swimmer_test = H5PYDataset('/Tmp/alitaiga/sim-to-real/swimmer_h{}_random.h5'.format(history), which_sets=('valid',),
        sources=('previous_obs','actions','observation_sim', 'observation_real'))

train_stream = DataStream(
    swimmer_train,
    iteration_scheme=ShuffledScheme(swimmer_train.num_examples, BATCH)
)

test_stream = DataStream(
    swimmer_test,
    iteration_scheme=ShuffledScheme(swimmer_test.num_examples, BATCH)
)

extensions = [
    #Timing(),
    FinishAfter(after_n_epochs=150),
    # WeightClipping(parameters=generative_model.discriminator_parameters, after_batch=True),
    TrainingDataMonitoring(
        [loss, percent],  # model.outputs+auxiliary_variables,
        after_epoch=True),
    DataStreamMonitoring(
        [loss, percent, percent_sim],  # auxiliary_variables,
        test_stream,
        prefix="test"),
    Checkpoint('/Tmp/alitaiga/sim-to-real/gm_his{}_h{}_bn.tar'.format(history, h), every_n_epochs=15, after_training=True,
               use_cpickle=True),
    ProgressBar(),
    Printing(),
]

main_loop = MainLoop(
    algorithm,
    data_stream=train_stream,
    model=model,
    extensions=extensions,
    # generator_algorithm=generator_algo
)

main_loop.run()
