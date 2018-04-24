import gym
import gym_throwandpush
from gym.monitoring import VideoRecorder
import numpy as np

from torch.utils.data import DataLoader

from simple_joints_lstm.pusher_lstm import LstmSimpleNet2Pusher
from simple_joints_lstm.mujoco_traintest_dataset_pusher_simple import MujocoTraintestPusherSimpleDataset

BATCH_SIZE = 1
ACTION_STEPS = 5
DATASET_PATH = "../data-collection/mujoco_pusher3dof_simple_{}act.npz".format(ACTION_STEPS)


env_sim = gym.make('Pusher3Dof2-v0')  # sim
env_real = gym.make('Pusher3Dof2-v0')  # real

env_sim.env._init(  # sim
    torques=[1, 1, 1],
    colored=True
)
env_sim.reset()

env_real.env._init(  # real
    torques=[1, 1, 1],
    xml='3link_gripper_push_2d_backlash',
    colored=False
)
env_real.reset()


def match_env(real, sim):
    # set env1 (simulator) to that of env2 (real robot)
    sim.env.set_state(
        real.env.model.data.qpos.ravel(),
        real.env.model.data.qvel.ravel()
    )



dataset_train = MujocoTraintestPusherSimpleDataset(DATASET_PATH, for_training=True)
dataloader_train = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True)

match_env(env_real, env_sim)
video_recorder = VideoRecorder(
    env_real, 'real.mp4', enabled=True)
video_recorder2 = VideoRecorder(
    env_sim, 'sim.mp4', enabled=True)

for i, data in enumerate(dataloader_train):
    for j in range(50):
        env_sim.render()
        env_real.render()

        action = data["actions"][0, j].numpy()

        video_recorder.capture_frame()
        video_recorder2.capture_frame()

        obs_real, _, _, _ = env_real.step(action.copy())
        obs_simp, _, _, _ = env_sim.step(action.copy())


    env_real.reset()
    env_sim.reset()
    match_env(env_real, env_sim)
    if i == 10:
        break

video_recorder.close()
video_recorder.enabled = False
video_recorder2.close()
video_recorder2.enabled = False
