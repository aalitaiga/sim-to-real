import argparse

import gym
import gym_reacher2
import numpy as np
from tqdm import tqdm
from ddpg.ddpg import DDPG
from ddpg.evaluator import Evaluator
from ddpg.main import train, test
from ddpg.normalized_env import NormalizedEnv
from ddpg.util import *

try:
    from hyperdash import Experiment

    hyperdash_support = True
except:
    hyperdash_support = False

parser = argparse.ArgumentParser(description='PyTorch DDPG Reacher2-v1')

parser.add_argument('--mode', default='train', type=str, help='support option: train/test')
parser.add_argument('--env', default='Reacher2-v1', type=str, help='open-ai gym environment')
parser.add_argument('--hidden1', default=400, type=int, help='hidden num of first fully connect layer')
parser.add_argument('--hidden2', default=300, type=int, help='hidden num of second fully connect layer')
parser.add_argument('--rate', default=0.001, type=float, help='learning rate')
parser.add_argument('--prate', default=0.0001, type=float, help='policy net learning rate (only for DDPG)')
parser.add_argument('--warmup', default=100, type=int, help='time without training but only filling the replay memory')
parser.add_argument('--discount', default=0.99, type=float, help='')
parser.add_argument('--bsize', default=64, type=int, help='minibatch size')
parser.add_argument('--rmsize', default=6000000, type=int, help='memory size')
parser.add_argument('--window_length', default=1, type=int, help='')
parser.add_argument('--tau', default=0.001, type=float, help='moving average for target network')
parser.add_argument('--ou_theta', default=0.15, type=float, help='noise theta')
parser.add_argument('--ou_sigma', default=0.2, type=float, help='noise sigma')
parser.add_argument('--ou_mu', default=0.0, type=float, help='noise mu')
parser.add_argument('--validate_episodes', default=20, type=int, help='how many episode to perform during validate experiment')
parser.add_argument('--max_episode_length', default=500, type=int, help='')
parser.add_argument('--validate_steps', default=2000, type=int, help='how many steps to perform a validate experiment')
parser.add_argument('--output', default='rl-logs', type=str, help='')
parser.add_argument('--debug', dest='debug', action='store_true')
parser.add_argument('--init_w', default=0.003, type=float, help='')
parser.add_argument('--train_iter', default=200000, type=int, help='train iters each timestep')
parser.add_argument('--epsilon', default=50000, type=int, help='linear decay of exploration policy')
parser.add_argument('--seed', default=-1, type=int, help='')
parser.add_argument('--resume', default='default', type=str, help='Resuming model path for testing')
# parser.add_argument('--l2norm', default=0.01, type=float, help='l2 weight decay') # TODO
# parser.add_argument('--cuda', dest='cuda', action='store_true') # TODO

args = parser.parse_args()
args.output = get_output_folder(args.output, args.env)
if args.resume == 'default':
    args.resume = 'output/{}-run0'.format(args.env)

env = NormalizedEnv(gym.make(args.env))

env.env.env._init( #"simulator"
    arm0=.1,    # length of limb 1
    arm1=.1,     # length of limb 2
    torque0=1, # torque of joint 1
    torque1=3000,  # torque of joint 2
    topDown=False
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
    exp = Experiment("sim2real-ddpg-sim-reacher")
    for arg in ["env", "rate", "prate", "hidden1", "hidden2", "warmup", "discount",
                "bsize", "rmsize", "window_length", "tau", "ou_theta", "ou_sigma", "ou_mu",
                "validate_episodes", "max_episode_length", "validate_steps", "init_w",
                "train_iter", "epsilon", "seed", "resume"]:
        arg_val = getattr(args, arg)
        exp.param(arg, arg_val)

    train(args, args.train_iter, agent, env, evaluate,
        args.validate_steps, args.output, max_episode_length=args.max_episode_length, debug=args.debug, exp=exp)

    # when done
    exp.end()

elif args.mode == 'test':
    test(args.validate_episodes, agent, env, evaluate, args.resume,
        visualize=True, debug=args.debug)

else:
    raise RuntimeError('undefined mode {}'.format(args.mode))
