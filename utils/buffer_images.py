"""This class stores all of the samples for training.  It is able to
construct randomly selected batches of phi's from the stored history.
"""
import math
import pickle

from blocks.extensions import SimpleExtension
from fuel.datasets.hdf5 import H5PYDataset
import h5py
import numpy as np

floatX = 'float32'

class FIFO(object):
    def __init__(self, max_len):
        self.queue = list()
        self.max_len = max_len

    def __len__(self):
        return len(self.queue)

    def push(self, elem):
        self.queue.append(elem)
        self.queue = self.queue[-self.max_len:]

    def copy(self):
        return np.vstack(self.queue)

    def clear(self):
        self.queue = list()

class BufferImages(object):
    """A replay memory consisting of circular buffers for observed images,
actions, and rewards.

    """
    def __init__(self, image_dim, observation_dim, action_dim,
            rng, max_steps=100000):
        """Construct a DataSet.

        Arguments:
            observation_dim - dimension of the observation space (DoF)
            action_dim - dimension of the action space
            max_steps - the number of time steps to store
            phi_length - number of images to concatenate into a state
            rng - initialized numpy random number generator, used to
            choose random minibatches

        """

        # Store arguments.
        self.image_dim = image_dim
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.max_steps = max_steps
        self.rng = rng

        # Allocate the circular buffers and indices.
        self.images = np.zeros((max_steps,) + image_dim, dtype='uint8')
        self.observations = np.zeros((max_steps, observation_dim), dtype=floatX)
        self.actions = np.zeros((max_steps, action_dim), dtype=floatX)
        self.s_transition_img = np.zeros((max_steps,) + image_dim, dtype='uint8')
        self.r_transition_img = np.zeros((max_steps,) + image_dim, dtype='uint8')
        self.s_transition_obs = np.zeros((max_steps, observation_dim), dtype=floatX)
        self.r_transition_obs = np.zeros((max_steps, observation_dim), dtype=floatX)
        self.s_rewards = np.zeros(max_steps, dtype=floatX)
        self.r_rewards = np.zeros(max_steps, dtype=floatX)
        # self.s_terminal = np.zeros(max_steps, dtype='bool')
        # self.r_terminal = np.zeros(max_steps, dtype='bool')

        self.bottom = 0
        self.top = 0
        self.size = 0

    def add_sample(self, img, obs, action, s_transition_img, r_transition_img,
            s_transition_obs, r_transition_obs, s_reward, r_reward):
        """Add a time step record.

        Arguments:
            observation -- observed state
            action -- action chosen by the agent
            s_transition -- observed transition state in the simulator
            r_transition -- observed transition state in the real world
            s_rewards-- reward received after taking the action in the simulator
            r_reward -- reward received after taking the action in the real world
            s_terminal -- boolean indicating whether the episode ended in the simulator
            r_terminal -- boolean indicating whether the episode ended in the real world
        """
        self.images[self.top] = img
        self.observations[self.top] = obs
        self.actions[self.top] = action
        self.s_transition_img[self.top] = s_transition_img
        self.r_transition_img[self.top] = r_transition_img
        self.s_transition_obs[self.top] = s_transition_obs
        self.r_transition_obs[self.top] = r_transition_obs
        self.s_rewards[self.top] = s_reward
        self.r_rewards[self.top] = r_reward
        # self.s_terminal[self.top] = s_terminal
        # self.r_terminal[self.top] = r_terminal

        if self.size == self.max_steps:
            self.bottom = (self.bottom + 1) % self.max_steps
        else:
            self.size += 1
        self.top = (self.top + 1) % self.max_steps

    def __len__(self):
        """Return an approximate count of stored state transitions."""
        return max(0, self.size)

    def __bool__(self):
        return True

    @property
    def full(self):
        return self.size == self.max_steps

    def random_batch(self, batch_size):
        """Return corresponding observations, actions, rewards, and terminal status for
        batch_size randomly chosen state transitions.

        """
        # Allocate the response.
        images = np.zeros((batch_size,) + self.image_dim, dtype='uint8')
        observations = np.zeros((batch_size, self.observation_dim), dtype=floatX)
        actions = np.zeros((batch_size, self.action_dim), dtype=floatX)
        s_transition_img = np.zeros((batch_size,) + self.image_dim, dtype='uint8')
        r_transition_img = np.zeros((batch_size,) + self.image_dim, dtype='uint8')
        s_transition_obs = np.zeros((batch_size, self.observation_dim), dtype=floatX)
        r_transition_obs = np.zeros((batch_size, self.observation_dim), dtype=floatX)
        s_rewards = np.zeros((batch_size), dtype=floatX)
        r_rewards = np.zeros((batch_size), dtype=floatX)
        # terminal = np.zeros((batch_size, 1), dtype='bool')

        count = 0
        while count < batch_size:
            # Randomly choose a time step from the replay memory.
            index = self.rng.randint(0, self.size)

            # Add the state transition to the response.
            images[count] = self.images.take(index, axis=0, mode='wrap')
            observations[count] = self.observations.take(index, axis=0, mode='wrap')
            actions[count] = self.actions.take(index, mode='wrap')
            s_transition_img[count] = self.s_transition_img.take(index, mode='wrap')
            r_transition_img[count] = self.r_transition_img.take(index, mode='wrap')
            s_transition_obs[count] = self.s_transition_obs.take(index, mode='wrap')
            r_transition_obs[count] = self.r_transition_obs.take(index, mode='wrap')
            s_rewards[count] = self.s_rewards.take(index, mode='wrap')
            r_rewards[count] = self.r_rewards.take(index, mode='wrap')
            # terminal[count] = self.terminal.take(end_index, mode='wrap')
            count += 1

        return {
            'images': images,
            'observations': observations,
            'actions': actions,
            's_transition_img': s_transition_img,
            'r_transition_img': r_transition_img,
            's_transition_obs': s_transition_obs,
            'r_transition_obs': r_transition_obs,
            's_rewards': s_rewards,
            'r_rewards': r_rewards
        }

    def save(self, path=None):
        """ Save the current data to disk """
        path = path or '/Tmp/alitaiga/sim-to-real/gen_model_data'
        with open(path, 'wb+') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path):
        with open(path, 'rb') as f:
            buf = pickle.load(f)
        return buf

class SaveBuffer(SimpleExtension):
    """ Extension to save the buffer during training """
    def __init__(self, buffer_, path='/Tmp/alitaiga/sim-to-real/buffer-test', **kwargs):
        self.buffer_ = buffer_
        self.path = path
        super(SaveBuffer, self).__init__(**kwargs)

    def do(self, which_callback, *args):
        self.buffer_.save(self.path)
        print("Current buffer size: {}".format(self.buffer_.size))
        if self.buffer_.full:
            self.main_loop.log.current_row['training_finish_requested'] = True

def buffer_to_h5(buffer_, split=0.9, name='/Tmp/alitaiga/sim-to-real/gen_data.h5'):
    """ Convert a buffer object into a h5 dataset """
    assert 0 < split <= 1
    size_train = math.floor(buffer_.size * split)
    size_val = math.ceil(buffer_.size * (1 - split))

    image_dim = buffer_.image_dim
    observation_dim = buffer_.observation_dim
    action_dim = buffer_.action_dim

    f = h5py.File(name, mode='w')
    images = f.create_dataset('images', (size_train+size_val,) + image_dim, dtype='uint8')
    observations = f.create_dataset('obs', (size_train+size_val, observation_dim), dtype='float32')
    actions = f.create_dataset('actions', (size_train+size_val, action_dim), dtype='float32')
    s_transition_img = f.create_dataset('s_transition_img', (size_train+size_val,) + image_dim, dtype='uint8')
    r_transition_img = f.create_dataset('r_transition_img', (size_train+size_val) + image_dim, dtype='uint8')
    s_transition_obs = f.create_dataset('s_transition_obs', (size_train+size_val, observation_dim), dtype='float32')
    r_transition_obs = f.create_dataset('r_transition_obs', (size_train+size_val, observation_dim), dtype='float32')
    reward_sim = f.create_dataset('reward_sim', (size_train+size_val,), dtype='float32')
    reward_real = f.create_dataset('reward_real', (size_train+size_val,), dtype='float32')

    split_dict = {
        'train': {
            'images': (0, size_train),
            'observations': (0, size_train),
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
            'observations': (size_train, size_train+size_val),
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

    train_indexs = np.random.choice(buffer_.size, size_train, replace=False)
    val_indexs = np.setdiff1d(np.arange(buffer_.size), train_indexs)

    images[:size_train, :, :] = buffer_.images[train_indexs, :, :]
    images[size_train:, :, :] = buffer_.images[val_indexs, :, :]
    observations[:size_train, :, :] = buffer_.observations[train_indexs, :, :]
    observations[size_train:, :, :] = buffer_.observations[val_indexs, :, :]
    f.flush()
    actions[:size_train, :, :] = buffer_.actions[train_indexs, :, :]
    actions[size_train:, :, :] = buffer_.actions[val_indexs, :, :]
    s_transition_img[:size_train, :] = buffer_.s_transition_img[train_indexs, :]
    s_transition_img[size_train:, :] = buffer_.s_transition_img[val_indexs, :]
    r_transition_img[:size_train, :] = buffer_.r_transition_img[train_indexs, :]
    r_transition_img[size_train:, :] = buffer_.r_transition_img[val_indexs, :]
    f.flush()
    s_transition_obs[:size_train, :] = buffer_.s_transition_obs[train_indexs, :]
    s_transition_obs[size_train:, :] = buffer_.s_transition_obs[val_indexs, :]
    r_transition_obs[:size_train, :] = buffer_.r_transition_obs[train_indexs, :]
    r_transition_obs[size_train:, :] = buffer_.r_transition_obs[val_indexs, :]
    reward_sim[:size_train] = buffer_.s_rewards[train_indexs]
    reward_sim[size_train:] = buffer_.s_rewards[val_indexs]
    reward_real[:size_train] = buffer_.r_rewards[train_indexs]
    reward_real[size_train:] = buffer_.r_rewards[val_indexs]
    f.flush()
    f.close()

    print('Created h5 dataset {}, with {} elements'.format(name, size_train+size_val))
    return f

if __name__ == '__main__':
    buf = BufferImages.load('/Tmp/alitaiga/sim-to-real/buffer-test')
    buffer_to_h5(buf)
