import torch
import gym
import gym_reacher2
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

args = get_args(env="Reacher-v0")

env = NormalizedEnv(gym.make(args.env))

if args.seed > 0:
    np.random.seed(args.seed)
    env.seed(args.seed)

nb_states = env.observation_space.shape[0]
nb_actions = env.action_space.shape[0]


agent = DDPG(nb_states, nb_actions, args)
evaluate = Evaluator(args.validate_episodes,
    args.validate_steps, args.output, max_episode_length=args.max_episode_length)

exp = None

if args.mode == 'train':
    if hyperdash_support:
        exp = Experiment("s2r-reacher-ddpg-real")
        import socket

        exp.param("host", socket.gethostname())
        exp.param("folder",args.output)
        for arg in ["env", "max_episode_length", "train_iter", "seed", "resume"]:
            arg_val = getattr(args, arg)
            exp.param(arg, arg_val)

    train(args, args.train_iter, agent, env, evaluate,
        args.validate_steps, args.output,
          max_episode_length=args.max_episode_length, debug=args.debug, exp=exp)

    # when done
    exp.end()

if args.mode == 'test':
    test(args.validate_episodes, agent, env, evaluate, args.resume,
         visualize=args.vis, debug=args.debug, load_best=args.best)

else:
    raise RuntimeError('undefined mode {}'.format(args.mode))
