import math

import h5py
from fuel.datasets.hdf5 import H5PYDataset
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

image_dim = (128, 128, 3)
observation_dim = int(env.observation_space[0].shape[0])
action_dim = int(env.action_space.shape[0])
rng = np.random.RandomState(seed=22)
max_steps = 110000
split = 0.91

# Creating the h5 dataset
name = '/Tmp/alitaiga/sim-to-real/gen_data.h5'
assert 0 < split <= 1
size_train = math.floor(max_steps * split)
size_val = math.ceil(max_steps * (1 - split))
f = h5py.File(name, mode='w')
images = f.create_dataset('images', (size_train+size_val,) + image_dim, dtype='uint8')
observations = f.create_dataset('obs', (size_train+size_val, observation_dim), dtype='float32')
actions = f.create_dataset('actions', (size_train+size_val, action_dim), dtype='float32')
s_transition_img = f.create_dataset('s_transition_img', (size_train+size_val,) + image_dim, dtype='uint8')
r_transition_img = f.create_dataset('r_transition_img', (size_train+size_val,) + image_dim, dtype='uint8')
s_transition_obs = f.create_dataset('s_transition_obs', (size_train+size_val, observation_dim), dtype='float32')
r_transition_obs = f.create_dataset('r_transition_obs', (size_train+size_val, observation_dim), dtype='float32')
reward_sim = f.create_dataset('reward_sim', (size_train+size_val,), dtype='float32')
reward_real = f.create_dataset('reward_real', (size_train+size_val,), dtype='float32')

split_dict = {
    'train': {
        'images': (0, size_train),
        'obs': (0, size_train),
        'actions': (0, size_train),
        's_transition_img': (0, size_train),
        'r_transition_img': (0, size_train),
        's_transition_obs': (0, size_train),
        'r_transition_obs': (0, size_train),
        'reward_sim': (0, size_train),
        'reward_real': (0, size_train)
    },
    'valid': {
        'images': (size_train, size_train+size_val),
        'obs': (size_train, size_train+size_val),
        'actions': (size_train, size_train+size_val),
        's_transition_img': (size_train, size_train+size_val),
        'r_transition_img': (size_train, size_train+size_val),
        's_transition_obs': (size_train, size_train+size_val),
        'r_transition_obs': (size_train, size_train+size_val),
        'reward_sim': (size_train, size_train+size_val),
        'reward_real': (size_train, size_train+size_val),
    }
}
f.attrs['split'] = H5PYDataset.create_split_array(split_dict)

def match_env(ev1, ev2):
    # make env1 state match env2 state (simulator matches real world)
    ev1.env.env.set_state(
        ev2.env.env.model.data.qpos.ravel(),
        ev2.env.env.model.data.qvel.ravel()
    )

i = 0

while i < max_steps:
    obs = env.reset()
    obs2 = env2.reset()
    match_env(env, env2)

    for t in range(150):
        # env.render()
        # env2.render()

        action = env.action_space.sample()
        new_obs, reward, done, info = env.step(action)
        new_obs2, reward2, done2, info2 = env2.step(action)

        images[i, :, :, :] = imresize(obs[1], [128, 128, 3])
        observations[i, :] = obs[0]
        actions[i, :] = action
        s_transition_img[i, :, :, :] = imresize(new_obs[1], [128, 128, 3])
        r_transition_img[i, :, :, :] = imresize(new_obs2[1], [128, 128, 3])
        s_transition_obs[i, :] = new_obs[0]
        r_transition_obs[i, :] = new_obs2[0]
        reward_sim[i] = reward
        reward_real[i] = reward2

        match_env(env, env2)
        i += 1

        if i % 5000 == 0:
            print("Buffer currently filled at: {}%".format(int(i*100./max_steps)))

        if i % 1500 == 0:
            f.flush()

        if done2:
            # print("Episode finished after {} timesteps".format(t+1))
            break

f.close()
print('Created h5 dataset with {} elements'.format(max_steps))
