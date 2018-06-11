from ddpg.args import Args
from reacher.run import run_reacher

ddpg_args = Args()

ddpg_args.parser.add_argument('--type', default="real", type=str, help='sim or real experiment')
ddpg_args.parser.add_argument('--t0', default=200, type=int, help='torque 1')
ddpg_args.parser.add_argument('--t1', default=200, type=int, help='torque 2')

args = ddpg_args.get_args(env="Reacher2-v0")

is_sim = False
if args.type == "sim":
    is_sim = True

run_reacher(args, sim=is_sim)

