"""This class stores all of the samples for training.  It is able to
construct randomly selected batches of phi's from the stored history.
"""
import pickle

import numpy as np
import theano

floatX = theano.config.floatX

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

class Buffer(object):
    """A replay memory consisting of circular buffers for observed images,
actions, and rewards.

    """
    def __init__(self, observation_dim, action_dim, rng, history=2, max_steps=1000):
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
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.max_steps = max_steps
        self.history = history
        # self.phi_length = phi_length
        self.rng = rng

        # Allocate the circular buffers and indices.
        self.observations = np.zeros((max_steps, history, observation_dim), dtype=floatX)
        self.actions = np.zeros((max_steps, history+1, action_dim), dtype=floatX)
        self.s_transition = np.zeros((max_steps, observation_dim), dtype=floatX)
        self.r_transition = np.zeros((max_steps, observation_dim), dtype=floatX)
        self.s_rewards = np.zeros(max_steps, dtype=floatX)
        self.r_rewards = np.zeros(max_steps, dtype=floatX)
        # self.s_terminal = np.zeros(max_steps, dtype='bool')
        # self.r_terminal = np.zeros(max_steps, dtype='bool')

        self.bottom = 0
        self.top = 0
        self.size = 0

    def add_sample(self, observation, action, s_transition, r_transition, s_reward, r_reward):  #, s_terminal, r_terminal):
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
        self.observations[self.top] = observation
        self.actions[self.top] = action
        self.s_transition[self.top] = s_transition
        self.r_transition[self.top] = r_transition
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

    def random_batch(self, batch_size):
        """Return corresponding observations, actions, rewards, and terminal status for
batch_size randomly chosen state transitions.

        """
        # Allocate the response.
        observations = np.zeros((batch_size, self.history, self.observation_dim),dtype=floatX)
        actions = np.zeros((batch_size, self.history+1, self.action_dim), dtype=floatX)
        s_transition = np.zeros((batch_size, self.observation_dim),dtype=floatX)
        r_transition = np.zeros((batch_size, self.observation_dim),dtype=floatX)
        s_rewards = np.zeros((batch_size, 1), dtype=floatX)
        r_rewards = np.zeros((batch_size, 1), dtype=floatX)
        # terminal = np.zeros((batch_size, 1), dtype='bool')

        count = 0
        while count < batch_size:
            # Randomly choose a time step from the replay memory.
            index = self.rng.randint(0, self.size)

            # Add the state transition to the response.
            observations[count] = self.observations.take(index, axis=0, mode='wrap')
            actions[count] = self.actions.take(index, mode='wrap')
            s_transition[count] = self.s_transition.take(index, mode='wrap')
            r_transition[count] = self.r_transition.take(index, mode='wrap')
            s_rewards[count] = self.s_rewards.take(index, mode='wrap')
            r_rewards[count] = self.r_rewards.take(index, mode='wrap')
            # terminal[count] = self.terminal.take(end_index, mode='wrap')
            count += 1

        return observations, actions, s_transition, r_transition, s_rewards, r_rewards

    def save(self, path=None):
        """ Save the current data to disk """
        path = path or '/Tmp/alitaiga/sim-to-real/buffer-test'
        with open(path, 'wb+') as f:
            pickle.dump(self.__dict__, f, pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path):
        buf = cls(0,0,0)
        with open(path, 'rb') as f:
            buf.__dict__ = pickle.load(f)
        return buf
