import torch
import gym
import gym_throwandpush
import numpy as np
from ddpg.ddpg import DDPG
from ddpg.evaluator import Evaluator
from ddpg.main import train, test
from ddpg.normalized_env import NormalizedEnv

from args.ddpg import get_args

try:
    from hyperdash import Experiment

    hyperdash_support = True
except:
    hyperdash_support = False

args = get_args(env="HalfCheetah2-v0")

env = NormalizedEnv(gym.make(args.env))

env.env.env._init( # real robot
    torques={
        "bthigh": 120,
        "bshin": 90,
        "bfoot": 60,
        "fthigh": 120,
        "fshin": 60,
        "ffoot": 30
    },
    colored=False
)

if args.seed > 0:
    np.random.seed(args.seed)
    env.seed(args.seed)

nb_states = env.observation_space.shape[0]
nb_actions = env.action_space.shape[0]


agent = DDPG(nb_states, nb_actions, args)
evaluate = Evaluator(args.validate_episodes,
    args.validate_steps, args.output, max_episode_length=args.max_episode_length)

exp = None

if args.mode == 'test':
    test(args.validate_episodes, agent, env, evaluate, args.resume,
         visualize=args.vis, debug=args.debug)

else:
    raise RuntimeError('undefined mode {}'.format(args.mode))
