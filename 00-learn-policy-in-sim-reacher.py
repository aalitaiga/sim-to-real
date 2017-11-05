import torch
import gym
import gym_reacher2
import numpy as np
from ddpg.args import Args
from tqdm import tqdm
from ddpg.ddpg import DDPG
from ddpg.evaluator import Evaluator
from ddpg.main import train, test
from ddpg.normalized_env import NormalizedEnv

try:
    from hyperdash import Experiment

    hyperdash_support = True
except:
    hyperdash_support = False

ddpg_args = Args()

ddpg_args.parser.add_argument('--t0', default=200, type=int, help='torque 1')
ddpg_args.parser.add_argument('--t1', default=200, type=int, help='torque 2')

args = ddpg_args.get_args(env="Reacher2-v0")

env = NormalizedEnv(gym.make(args.env))

env.env.env._init(  # real robot
    torque0=args.t0,  # torque of joint 1
    torque1=args.t1,  # torque of joint 2
    colors={
        "arenaBackground": ".27 .27 .81",
        "arenaBorders": "1.0 0.8 0.4",
        "arm0": "0.9 0.6 0.9",
        "arm1": "0.9 0.9 0.6"
    },
    topDown=True
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

if args.mode == 'train':
    if hyperdash_support:
        exp = Experiment("s2r-reacher-ddpg-real")
        import socket

        exp.param("host", socket.gethostname())
        exp.param("torques", [args.t0, args.t1])
        exp.param("folder", args.output)

        for arg in ["env", "max_episode_length", "train_iter", "seed", "resume"]:
            arg_val = getattr(args, arg)
            exp.param(arg, arg_val)

    train(args, args.train_iter, agent, env, evaluate,
          args.validate_steps, args.output,
          max_episode_length=args.max_episode_length, debug=args.debug, exp=exp)

    # when done
    exp.end()

elif args.mode == 'test':
    test(args.validate_episodes, agent, env, evaluate, args.resume,
         visualize=args.vis, debug=args.debug, load_best=args.best)

else:
    raise RuntimeError('undefined mode {}'.format(args.mode))
