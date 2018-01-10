from ddpg.args import Args

from pusher.run import run_pusher

ddpg_args = Args()

ddpg_args.parser.add_argument('--type', default="real", type=str, help='sim or real experiment')
ddpg_args.parser.add_argument('--t0', default=1, type=int, help='torque 1 - r_shoulder_pan_joint')
ddpg_args.parser.add_argument('--t1', default=1, type=int, help='torque 2 - r_shoulder_lift_joint')
ddpg_args.parser.add_argument('--t2', default=1, type=int, help='torque 3 - r_upper_arm_roll_joint')
ddpg_args.parser.add_argument('--t3', default=1, type=int, help='torque 4 - r_elbow_flex_joint')
ddpg_args.parser.add_argument('--t4', default=1, type=int, help='torque 5 - r_forearm_roll_joint')
ddpg_args.parser.add_argument('--t5', default=1, type=int, help='torque 6 - r_wrist_flex_joint')
ddpg_args.parser.add_argument('--t6', default=1, type=int, help='torque 7 - r_wrist_roll_joint')

args = ddpg_args.get_args(env="Pusher2-v0")

args.max_episode_length = 100 # gym standard for this env

is_sim = False
if args.type == "sim":
    is_sim = True

run_pusher(args, sim=is_sim)

