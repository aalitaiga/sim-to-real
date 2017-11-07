from ddpg.args import Args

from pusher3dof.run import run_pusher3dof

ddpg_args = Args()

ddpg_args.parser.add_argument('--type', default="real", type=str, help='sim or real experiment')
ddpg_args.parser.add_argument('--t0', default=1, type=float, help='torque 1 - proximal_j_1')
ddpg_args.parser.add_argument('--t1', default=1, type=float, help='torque 2 - distal_j_1')
ddpg_args.parser.add_argument('--t2', default=1, type=float, help='torque 3 - distal_j_2')

args = ddpg_args.get_args(env="Pusher3Dof2-v0")

args.max_episode_length = 100 # gym standard for this env (assumed)

is_sim = False
if args.type == "sim":
    is_sim = True

run_pusher3dof(args, sim=is_sim)

