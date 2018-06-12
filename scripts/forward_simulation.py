import gym
import gym_throwandpush
from gym.monitoring import VideoRecorder
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns; sns.set()
from pylab import rcParams

from fuel.streams import DataStream
from fuel.schemes import SequentialScheme
from fuel.datasets.hdf5 import H5PYDataset

from simple_joints_lstm.pusher_lstm import LstmSimpleNet2Pusher

plt.rc("axes.spines", top=False, right=False, bottom=True)
sns.set_style("white")
rcParams['figure.figsize'] = 15, 9
# plt.style.use('ggplot')

np.random.seed(0)

env = gym.make("Pusher3Dof2-v0")
env.env._init(
    torques=[1, 1, 1],
    xml='3link_gripper_push_2d_backlash',
    colored=False
)
env.reset()

env2 = gym.make("Pusher3Dof2Plus-v0")
# model = "/u/alitaiga/repositories/sim-to-real/trained_models/old_models/adrien_lstm_pusher_3l_128_best.pt"
model = "/u/alitaiga/repositories/sim-to-real/trained_models/lstm_pusher_3l_128_2000_best.pt"
env2.load_model(LstmSimpleNet2Pusher(27, 6, use_cuda=False), model)
env2.env.env._init(
    torques=[1, 1, 1],
    colored=True,
)
env2.reset()

def match_env(env_1, env_2):
    # set env2 (simulator) to that of env1 (real robot)
    env_2.unwrapped.set_state(
        env_1.unwrapped.model.data.qpos.ravel(),
        env_1.unwrapped.model.data.qvel.ravel()
    )


# DATASET_PATH_REL = "/data/lisa/data/sim2real/"
# DATASET_PATH = DATASET_PATH_REL + "mujoco_data_pusher3dof_big_backl.h5"
# batch_size = 1
# train_data = H5PYDataset(
#     DATASET_PATH, which_sets=('train',), sources=('s_transition_obs','r_transition_obs', 'obs', 'actions')
# )
# stream_train = DataStream(train_data, iteration_scheme=SequentialScheme(train_data.num_examples, batch_size))
# valid_data = H5PYDataset(
#     DATASET_PATH, which_sets=('valid',), sources=('s_transition_obs','r_transition_obs', 'obs', 'actions')
# )
# stream_valid = DataStream(valid_data, iteration_scheme=SequentialScheme(train_data.num_examples, batch_size))

# iterator = stream_train.get_epoch_iterator(as_dict=True)
# data = next(iterator)

# length = data["actions"].shape[1]
length = 10

video_recorder = VideoRecorder(
    env, 'real_backlash.mp4', enabled=False)
video_recorder2 = VideoRecorder(
    env2, 'sim+.mp4', enabled=False)

# Only first six predicted by the lstm
num_obs = 10
array = np.zeros((2, 100, num_obs))

for i, data in enumerate(range(1)):
    env.reset()
    env2.reset()
    match_env(env, env2)
    new_obs = env.unwrapped._get_obs()
    new_obs2 = env2.unwrapped._get_obs()

    for j in range(100):
        # env.render()
        # env2.render()
        # import ipdb; ipdb.set_trace()
        array[0, j, :] = new_obs[:num_obs]
        array[1, j, :] = new_obs2[:num_obs]

        # action = data["actions"][0,j,:]
        action = env.action_space.sample()
        # video_recorder.capture_frame()
        # video_recorder2.capture_frame()
        new_obs, reward, done, info = env.step(action.copy())
        new_obs2, reward2, done2, info2 = env2.step(action.copy())

        if done:
            break

    if i == 1:
        break

# video_recorder.close()
# video_recorder.enabled = False
# video_recorder2.close()
# video_recorder2.enabled = False
# colors = ['r', 'g', 'b', 'y', 'c', 'm', 'k']
# cm_name = 'Pastel1'
cm_name ='tab10'
color_map = cm.get_cmap(cm_name)
colors = color_map([x/(float(num_obs)+2) for x in range(num_obs)])

x = np.arange(j+1)
for y in range(num_obs):
    plt.plot(x, array[0,:,y], color=colors[y])
    plt.plot(x, array[1,:,y], linestyle='--', color=colors[y])
plt.title("Forward simulation on Pusher3Dof")
plt.savefig('forward_sim+_all.jpg', bbox_inches='tight')
plt.show()
