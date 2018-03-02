import gym
import gym_throwandpush
from gym.monitoring import VideoRecorder
import numpy as np

from torch.utils.data import DataLoader

from simple_joints_lstm.lstm_simple_net2_pusher import LstmSimpleNet2Pusher
from simple_joints_lstm.mujoco_traintest_dataset_pusher_simple import MujocoTraintestPusherSimpleDataset

BATCH_SIZE = 1
ACTION_STEPS = 5
MODEL_PREFIX = "/home/florian/dev/sim-to-real/trained_models/"
MODEL = MODEL_PREFIX + "lstm_pusher3dof_simple_{}ac_3l_128n_best.pt".format(ACTION_STEPS)
DATASET_PATH = "../data-collection/mujoco_pusher3dof_simple_{}act.npz".format(ACTION_STEPS)


env_real = gym.make("Pusher3Dof2-v0")
env_real.env._init(
    torques=[1, 1, 1],
    xml='3link_gripper_push_2d_backlash'
)
env_real.reset()

env_simplus = gym.make("Pusher3Dof2Plus2-v0")
env_simplus.load_model(LstmSimpleNet2Pusher(15, 6, use_cuda=False), MODEL)
env_simplus.env.env._init(
    torques=[1, 1, 1],
)
env_simplus.reset()


def match_env(real, sim):
    # set env1 (simulator) to that of env2 (real robot)
    sim.env.env.set_state(
        real.env.model.data.qpos.ravel(),
        real.env.model.data.qvel.ravel()
    )



dataset_train = MujocoTraintestPusherSimpleDataset(DATASET_PATH, for_training=True)
dataloader_train = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True)

match_env(env_real, env_simplus)
video_recorder = VideoRecorder(
    env_real, 'real.mp4', enabled=True)
video_recorder2 = VideoRecorder(
    env_simplus, 'sim+.mp4', enabled=True)

for i, data in enumerate(dataloader_train):
    for j in range(50):
        env_real.render()
        env_simplus.render()

        action = data["actions"][0, j].numpy()

        video_recorder.capture_frame()
        video_recorder2.capture_frame()

        obs_real, _, _, _ = env_real.step(action.copy())
        obs_simp, _, _, _ = env_simplus.step(action.copy())

        print (np.around(obs_real,2), np.around(obs_simp,2), np.around(action,2))
        print (" ")

    env_real.reset()
    env_simplus.reset()
    match_env(env_real, env_simplus)
    if i == 30:
        break

video_recorder.close()
video_recorder.enabled = False
video_recorder2.close()
video_recorder2.enabled = False
