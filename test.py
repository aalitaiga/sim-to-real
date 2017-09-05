import gym
import gym_reacher2
import numpy as np
from scipy.misc import imresize
from utils.buffer_images import BufferImages as Buffer
import matplotlib.pyplot as plt

env = gym.make('Reacher2Pixel-v1')
env2 = gym.make('Reacher2Pixel-v1')

env.env.env._init(
    arm0=.1,    # length of limb 1
    arm1=.1,     # length of limb 2
    torque0=200, # torque of joint 1
    torque1=200  # torque of joint 2
)
env2.env.env._init(
    arm0=.12,    # length of limb 1
    arm1=.08,     # length of limb 2
    torque0=200, # torque of joint 1
    torque1=200,  # torque of joint 2
    fov=50,
    colors={
        "arenaBackground": ".27 .27 .81",
        "arenaBorders": "1.0 0.8 0.4",
        "arm0": "0.2 0.6 0.2",
        "arm1": "0.2 0.6 0.2"
    }
)
# import ipdb; ipdb.set_trace()
image_dim = (128, 128, 3)
observation_dim = int(env.observation_space[0].shape[0])
action_dim = int(env.action_space.shape[0])
rng = np.random.RandomState(seed=22)
max_steps = 100
buffer_ = Buffer(image_dim, observation_dim, action_dim, rng, max_steps)

def match_env(ev1, ev2):
    # make env1 state match env2 state (simulator matches real world)
    ev1.env.env.set_state(
        ev2.env.env.model.data.qpos.ravel(),
        ev2.env.env.model.data.qvel.ravel()
    )

s = 0
while not buffer_.full:
    obs = env.reset()
    obs2 = env2.reset()
    match_env(env, env2)

    print("Episode: {}".format(s))
    for t in range(150):
        # env.render()
        # env2.render()

        action = env.action_space.sample()
        new_obs, reward, done, info = env.step(action)
        new_obs2, reward2, done2, info2 = env2.step(action)
        print(env.env.env.model.data.qpos.flat[2:])

        # plt.imshow(imresize(obs[1], [128, 128, 3]))
        # import ipdb; ipdb.set_trace()
        buffer_.add_sample(
            imresize(obs[1], [128, 128, 3]), obs[0], action,
            imresize(new_obs[1], [128, 128, 3]), imresize(new_obs2[1], [128, 128, 3]),
            new_obs[0], new_obs2[0], reward, reward2)
        match_env(env, env2)
        # import ipdb; ipdb.set_trace()

        if len(buffer_) % 1000 == 0:
            print("Buffer currently filled at: {}%".format(int(len(buffer_)*100./max_steps)))

        if done:
            # print("Episode finished after {} timesteps".format(t+1))
            break
    s = s + 1

buffer_.save('/Tmp/alitaiga/sim-to-real/gen_model_data_{}'.format(max_steps))
