import gym
import gym_reacher2
import numpy as np
import scipy.misc
from utils.buffer_images import BufferImages as Buffer


env = gym.make('Reacher2Pixel-v1')
env2 = gym.make('Reacher2Pixel-v1')

env.env.env._init(
    arm0 = .1,    # length of limb 1
    arm1 = .1,     # length of limb 2
    torque0 = 200, # torque of joint 1
    torque1 = 200  # torque of joint 2
)
env2.env.env._init(
    arm0 = .05,    # length of limb 1
    arm1 = .15,     # length of limb 2
    torque0 = 200, # torque of joint 1
    torque1 = 200  # torque of joint 2
)

image_dim = (500, 500, 3) #env.observation_space[1]
observation_dim = int(env.observation_space[0].shape[0])
action_dim = int(env.action_space.shape[0])
rng = np.random.RandomState(seed=23)
max_steps = 10
buffer_ = Buffer(image_dim, observation_dim, action_dim, rng, max_steps)


def match_env(ev1, ev2):
    # make env1 state match env2 state (simulator matches real world)
    ev1.env.env.set_state(
        ev2.env.env.model.data.qpos.ravel(),
        ev2.env.env.model.data.qvel.ravel()
    )

# for i_episode in range(1000):
while not buffer_.full:
    obs = env.reset()
    obs2 = env2.reset()
    match_env(env, env2)

    for t in range(100):
        env.render()
        env2.render()

        action = env.action_space.sample()
        new_obs, reward, done, info = env.step(action)
        new_obs2, reward2, done2, info2 = env2.step(action)

        buffer_.add_sample(obs[1], obs[0], action, new_obs[1], new_obs2[1],
            new_obs[0], new_obs2[0], reward, reward2)
        match_env(env, env2)

        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break

buffer_.save()
