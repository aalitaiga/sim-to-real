import gym
import gym_throwandpush
from gym.monitoring import VideoRecorder

from fuel.streams import DataStream
from fuel.schemes import SequentialScheme
from fuel.datasets.hdf5 import H5PYDataset

from simple_joints_lstm.pusher_lstm import LstmSimpleNet2Pusher

env = gym.make("Pusher3Dof2-v0")
env.env._init(
    torques=[1, 1, 1],
    xml='3link_gripper_push_2d_backlash'
)
env.reset()

env2 = gym.make("Pusher3Dof2Plus-v0")
model = "/u/alitaiga/repositories/sim-to-real/trained_models/lstm_pusher_3l_128_best.pt"
env2.load_model(LstmSimpleNet2Pusher(27,6, cuda=False), model)
env2.env.env._init(
    torques=[1, 1, 1],
)
env2.reset()

def match_env(ev1, ev2):
    # set env1 (simulator) to that of env2 (real robot)
    ev1.env.env.set_state(
        ev2.env.env.model.data.qpos.ravel(),
        ev2.env.env.model.data.qvel.ravel()
    )

DATASET_PATH_REL = "/data/lisa/data/sim2real/"
DATASET_PATH = DATASET_PATH_REL + "mujoco_data_pusher3dof_5ac_backl.h5"
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
length = data["actions"].shape[1]

match_env(env, env2)
video_recorder = VideoRecorder(
    env, 'sim+backlash.mp4', enabled=True)
video_recorder2 = VideoRecorder(
    env2, 'sim+.mp4', enabled=True)

for i, data in enumerate(stream_train.get_epoch_iterator(as_dict=True)):
    for j in range(length):
    	action = data["actions"]
    	video_recorder.capture_frame()
    	video_recorder2.capture_frame()
    	new_obs, reward, done, info = env.step(action)
        new_obs2, reward2, done2, info2 = env2.step(action)

    if i == 4:
        break

video_recorder.close()
video_recorder.enabled = False
video_recorder2.close()
video_recorder2.enabled = False