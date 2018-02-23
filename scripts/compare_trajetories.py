import gym
import gym_throwandpush
from gym.monitoring import VideoRecorder

from fuel.streams import DataStream
from fuel.schemes import SequentialScheme
from fuel.datasets.hdf5 import H5PYDataset

from simple_joints_lstm.lstm_simple_net2_pusher import LstmSimpleNet2Pusher

env = gym.make("Pusher3Dof2-v0")
env.env._init(
    torques=[1, 1, 1],
    xml='3link_gripper_push_2d_backlash'
)
env.reset()

env2 = gym.make("Pusher3Dof2Plus-v0")
model = "/u/alitaiga/repositories/sim-to-real/trained_models/lstm_pusher_5ac_3l_128_best.pt"
env2.load_model(LstmSimpleNet2Pusher(15,6, use_cuda=False), model)
env2.env.env._init(
    torques=[1, 1, 1],
)
env2.reset()

def match_env(env_2, env_1):
    # set env1 (simulator) to that of env2 (real robot)
    env_1.env.env.set_state(
        env_2.env.model.data.qpos.ravel(),
        env_2.env.model.data.qvel.ravel()
    )

DATASET_PATH_REL = "/data/lisa/data/sim2real/"
DATASET_PATH = DATASET_PATH_REL + "mujoco_data_pusher3dof_big_backl.h5"
batch_size = 1
train_data = H5PYDataset(
    DATASET_PATH, which_sets=('train',), sources=('s_transition_obs','r_transition_obs', 'obs', 'actions')
)
stream_train = DataStream(train_data, iteration_scheme=SequentialScheme(train_data.num_examples, batch_size))
valid_data = H5PYDataset(
    DATASET_PATH, which_sets=('valid',), sources=('s_transition_obs','r_transition_obs', 'obs', 'actions')
)
stream_valid = DataStream(valid_data, iteration_scheme=SequentialScheme(train_data.num_examples, batch_size))

iterator = stream_train.get_epoch_iterator(as_dict=True)

data = next(iterator)
# import ipdb; ipdb.set_trace()
length = data["actions"].shape[1]

match_env(env, env2)
video_recorder = VideoRecorder(
    env, 'sim+backlash.mp4', enabled=True)
video_recorder2 = VideoRecorder(
    env2, 'sim+.mp4', enabled=True)

for i, data in enumerate(iterator):
    for j in range(length):
        env.render()
        env2.render()

        action = data["actions"][0,j,:]
        video_recorder.capture_frame()
        video_recorder2.capture_frame()
        new_obs, reward, done, info = env.step(action.copy())
        obs = new_obs[:6]
        new_obs2, reward2, done2, info2 = env2.step(action.copy())

    env.reset()
    env2.reset()
    match_env(env, env2)
    if i == 30:
        break

video_recorder.close()
video_recorder.enabled = False
video_recorder2.close()
video_recorder2.enabled = False
