import logging
import numpy as np

from blocks.algorithms import Adam, RMSProp
from blocks.bricks import MLP, Tanh, Identity, LeakyRectifier
from blocks.extensions import FinishAfter, Printing
from blocks.extensions.monitoring import TrainingDataMonitoring
from blocks.extensions.saveload import Checkpoint
from blocks.filter import VariableFilter
from blocks.initialization import IsotropicGaussian, Constant
from blocks.model import Model
from rllab.envs.gym_env import GymEnv
from rllab.envs.normalized_env import normalize
import theano.tensor as T

from buffer_ import Buffer

# logging.basicConfig(filename='example.log',
#     level=logging.DEBUG,
#     format='%(message)s')
#
# logging.getLogger().addHandler(logging.StreamHandler())
# logger = logging.getLogger(__name__)

env = normalize(GymEnv('Swimmer-v1', force_reset=True), normalize_obs=True)

## Defining the buffer
buffer_ = Buffer.load('/Tmp/alitaiga/sim-to-real/buffer-test')

observation_dim = buffer_.observation_dim
action_dim = buffer_.action_dim
history = buffer_.history
rng = np.random.RandomState(seed=23)


## Defining the predictive model
input_dim = (history+1)*observation_dim + history*action_dim
h = 128
LEARNING_RATE = 5e-5
BETA1 = 0.5

# Doesn't make sense to use batch_norm in online training
# so far the environnement and generative model run synchronously
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

G = MLP(
    activations=[LeakyRectifier(), LeakyRectifier(), Tanh()],
    dims=[input_dim, h, h, observation_dim],
    name='generator'
)
D = MLP(
    activations=[LeakyRectifier(), LeakyRectifier(), Identity()],
    dims=[input_dim, h, h, 1],
    name='discriminator'
)
