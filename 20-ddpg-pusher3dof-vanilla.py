from ddpg.args import Args

from pusher3dof.run import run_pusher3dof

ddpg_args = Args()

args = ddpg_args.get_args(env="Pusher3Dof-v0")

args.max_episode_length = 100 # gym standard for this env (assumed)

run_pusher3dof(args, sim=False, vanilla=True)

