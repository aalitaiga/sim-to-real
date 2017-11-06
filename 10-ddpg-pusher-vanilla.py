from ddpg.args import Args

from pusher.run import run_pusher

ddpg_args = Args()

args = ddpg_args.get_args(env="Pusher-v0")

args.max_episode_length = 100 # gym standard for this env

run_pusher(args, sim=False, vanilla=True)

