import gym
import gym_throwandpush
import numpy as np
from ddpg.args import Args
from ddpg.ddpg import DDPG
from ddpg.evaluator import Evaluator
from ddpg.main import train, test
from ddpg.normalized_env import NormalizedEnv


def run_pusher(args, sim=True, vanilla=False):
    try:
        from hyperdash import Experiment

        hyperdash_support = True
    except:
        hyperdash_support = False

    env = NormalizedEnv(gym.make(args.env))

    torques = [1.0] * 7  # if real
    colored = False
    topDown = False

    if sim:
        torques = [args.t0, args.t1, args.t2, args.t3, args.t4, args.t5, args.t6]
        topDown = True,
        colored = True

    if not vanilla:
        env.env.env._init(  # real robot
            torques={
                "r_shoulder_pan_joint": torques[0],
                "r_shoulder_lift_joint": torques[1],
                "r_upper_arm_roll_joint": torques[2],
                "r_elbow_flex_joint": torques[3],
                "r_forearm_roll_joint": torques[4],
                "r_wrist_flex_joint": torques[5],
                "r_wrist_roll_joint": torques[6]
            },
            topDown=topDown,
            colored=colored
        )

    if args.seed > 0:
        np.random.seed(args.seed)
        env.seed(args.seed)

    nb_states = env.observation_space.shape[0]
    nb_actions = env.action_space.shape[0]

    agent = DDPG(nb_states, nb_actions, args)
    evaluate = Evaluator(
        args.validate_episodes,
        args.validate_steps,
        args.output,
        max_episode_length=args.max_episode_length
    )

    exp = None

    if args.mode == 'train':
        if hyperdash_support:
            prefix = "real"
            if sim: prefix = "sim"

            exp = Experiment("s2r-pusher-ddpg-{}".format(prefix))
            import socket

            exp.param("host", socket.gethostname())
            exp.param("type", prefix)  # sim or real
            exp.param("vanilla", vanilla)  # vanilla or not
            exp.param("torques", torques)
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
