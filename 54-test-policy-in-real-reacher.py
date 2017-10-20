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

args = get_args(env="Reacher2-v1")


# # to run eval:
# # python3 54-test-policy-in-real-reacher.py --mode test --debug --resume rl-logs/Reacher2-v1-run3 --vis
# # or
# # python3 54-test-policy-in-real-reacher.py --mode test --debug --resume rl-logs/Reacher2Plus-v1-run3 --vis


env = NormalizedEnv(gym.make(args.env))

env.env.env._init( # real robot
    torque0=200, # torque of joint 1
    torque1=200,  # torque of joint 2
    colors={
        "arenaBackground": ".27 .27 .81",
        "arenaBorders": "1.0 0.8 0.4",
        "arm0": "0.9 0.6 0.9",
        "arm1": "0.9 0.9 0.6"
    },
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

# # TRAIN MODE NEVER HAPPENS ON REAL ROBOT

# if args.mode == 'train':
#     if hyperdash_support:
#         exp = Experiment("sim2real-ddpg-simplus-reacher")
#         exp.param("model", MODEL_PATH)
#         for arg in ["env", "rate", "prate", "hidden1", "hidden2", "warmup", "discount",
#                     "bsize", "rmsize", "window_length", "tau", "ou_theta", "ou_sigma", "ou_mu",
#                     "validate_episodes", "max_episode_length", "validate_steps", "init_w",
#                     "train_iter", "epsilon", "seed", "resume"]:
#             arg_val = getattr(args, arg)
#             exp.param(arg, arg_val)
#
#     train(args, args.train_iter, agent, env, evaluate,
#         args.validate_steps, args.output, max_episode_length=args.max_episode_length, debug=args.debug, exp=exp)
#
#     # when done
#     exp.end()

if args.mode == 'test':
    test(args.validate_episodes, agent, env, evaluate, args.resume,
        visualize=args.vis, debug=args.debug)

else:
    raise RuntimeError('undefined mode {}'.format(args.mode))
