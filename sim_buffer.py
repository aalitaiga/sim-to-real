import logging
import numpy as np

from blocks.algorithms import Adam, RMSProp, GradientDescent
from blocks.bricks import MLP, Tanh, Identity, LeakyRectifier
from blocks.extensions import FinishAfter, Printing, ProgressBar
from blocks.extensions.monitoring import DataStreamMonitoring, TrainingDataMonitoring
from blocks.extensions.saveload import Checkpoint
from blocks.initialization import IsotropicGaussian, Constant
from blocks.model import Model
from fuel.streams import DataStream
from fuel.schemes import ShuffledScheme
from fuel.datasets.hdf5 import H5PYDataset
from rllab.envs.gym_env import GymEnv
from rllab.envs.normalized_env import normalize
import theano.tensor as T

from buffer_ import Buffer
from generative_model import WGAN, WeightClipping
from original_main_loop import MainLoop

# logging.basicConfig(filename='example.log',
#     level=logging.DEBUG,
#     format='%(message)s')
#
# logging.getLogger().addHandler(logging.StreamHandler())
# logger = logging.getLogger(__name__)

env = normalize(GymEnv('Swimmer-v1', force_reset=True), normalize_obs=True)

## Defining the buffer
buffer_ = Buffer.load('/Tmp/alitaiga/sim-to-real/buffer-test')
# dataset = buffer_to_h5(buffer_)

observation_dim = buffer_.observation_dim
action_dim = buffer_.action_dim
history = buffer_.history
rng = np.random.RandomState(seed=23)
del buffer_

## Defining the predictive model
input_dim = (history+1)*observation_dim + history*action_dim
h = 128
LEARNING_RATE = 5e-5
BETA1 = 0.5
BATCH = 32

# Doesn't make sense to use batch_norm in online training
# so far the environnement and generative model run synchronously
actions_var = env.action_space.new_tensor_variable(
    'actions',
    extra_dims=2
)
obs_prev = env.observation_space.new_tensor_variable(
    'observations',
    extra_dims=2
)
obs_sim = env.observation_space.new_tensor_variable(
    'observation_sim',
    extra_dims=1
)
obs_real = env.observation_space.new_tensor_variable(
    'observation_real',
    extra_dims=1
)
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

predictive_model = MLP(
    activations=[LeakyRectifier(), LeakyRectifier(), Tanh()],
    dims=[input_dim, h, h, observation_dim],
    weights_init=IsotropicGaussian(std=0.02, mean=0),
    biases_init=Constant(0.01),
    name='predictive_model'
)
predictive_model.initialize()
obs_predicted = predictive_model.apply(context)
loss = T.sqr(obs_predicted - obs_real).mean()
loss.name = 'squared_error'

percent = abs((obs_predicted - obs_real) / obs_real).mean()
percent.name = "percent_error"

model = Model(loss)
algorithm = GradientDescent(
    cost=loss,
    parameters=model.parameters,
    step_rule=Adam(learning_rate=LEARNING_RATE, beta1=BETA1),
    on_unused_sources='ignore',
    theano_func_kwargs={'allow_input_downcast': True}
)

# auxiliary_variables = [u for u in model.auxiliary_variables if 'percent_error' in u.name]
swimmer_train = H5PYDataset('/Tmp/alitaiga/sim-to-real/data_swimmer.h5', which_sets=('train',),
        sources=('observations','actions','observation_sim', 'observation_real'))
swimmer_test = H5PYDataset('/Tmp/alitaiga/sim-to-real/data_swimmer.h5', which_sets=('valid',),
        sources=('observations','actions','observation_sim', 'observation_real'))

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
    FinishAfter(after_n_epochs=100),
    # WeightClipping(parameters=generative_model.discriminator_parameters, after_batch=True),
    TrainingDataMonitoring(
        [loss, percent],  # model.outputs+auxiliary_variables,
        prefix="train",
        after_epoch=True),
    DataStreamMonitoring(
        [loss, percent],  # auxiliary_variables,
        test_stream,
        prefix="test"),
    Checkpoint('generative_model.tar', every_n_epochs=3, after_training=True,
               use_cpickle=True),
    Printing(),
    ProgressBar()
]

main_loop = MainLoop(
    algorithm,
    data_stream=train_stream,
    model=model,
    extensions=extensions,
    # generator_algorithm=generator_algo
)

main_loop.run()
