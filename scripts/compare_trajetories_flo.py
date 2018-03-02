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
MODEL = MODEL_PREFIX + "lstm_pusher3dof_simple_{}ac_3l_128n.pt".format(ACTION_STEPS)
DATASET_PATH = "../data-collection/mujoco_pusher3dof_simple_{}act.npz".format(ACTION_STEPS)

env_sim = gym.make('Pusher3Dof2-v0')  # sim
env_simplus = gym.make('Pusher3Dof2Plus2-v0')  # sim
env_real = gym.make('Pusher3Dof2-v0')  # real

env_sim.env._init(  # sim
    torques=[1, 1, 1],
    colored=True
)
env_sim.reset()

env_simplus.load_model(LstmSimpleNet2Pusher(15, 6, use_cuda=False, normalized=False), MODEL)
env_simplus.env.env._init(  # sim
    torques=[1, 1, 1],
    colored=True
)
env_simplus.reset()

env_real.env._init(  # real
    torques=[1, 1, 1],
    xml='3link_gripper_push_2d_backlash',
    colored=False
)
env_real.reset()


def match_env(real, sim):
    # set env1 (simulator) to that of env2 (real robot)
    simulator = sim.env
    if hasattr(simulator, "env"):
        simulator = simulator.env

    simulator.set_state(
        real.env.model.data.qpos.ravel(),
        real.env.model.data.qvel.ravel()
    )


dataset_train = MujocoTraintestPusherSimpleDataset(DATASET_PATH, for_training=True)
dataloader_train = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True)

match_env(env_real, env_simplus)
match_env(env_real, env_sim)
video_recorder_real = VideoRecorder(
    env_real, 'real.mp4', enabled=True)
video_recorder_simplus = VideoRecorder(
    env_simplus, 'sim+.mp4', enabled=True)
video_recorder_sim = VideoRecorder(
    env_sim, 'sim.mp4', enabled=True)

video_recorders = [video_recorder_real, video_recorder_simplus, video_recorder_sim]

for i, data in enumerate(dataloader_train):
    for j in range(50):
        env_real.render()
        env_simplus.render()
        env_sim.render()

        action = data["actions"][0, j].numpy()

        for vr in video_recorders:
            vr.capture_frame()

        obs_real, _, _, _ = env_real.step(action.copy())
        obs_simp, _, _, _ = env_simplus.step(action.copy())
        obs_sim, _, _, _ = env_sim.step(action.copy())

        # print (np.around(obs_real,2), np.around(obs_simp,2), np.around(action,2))
        # print (" ")

    env_real.reset()
    env_simplus.reset()
    env_sim.reset()
    match_env(env_real, env_simplus)
    match_env(env_real, env_sim)
    if i == 10:
        break

for vr in video_recorders:
    vr.close()
    vr.enabled = False

# ffmpeg \
#   -i real.mp4 \
#   -i sim+.mp4 \
#   -filter_complex '[0:v]pad=iw*2:ih[int];[int][1:v]overlay=W/2:0[vid]' \
#   -map "[vid]" \
#   -c:v libx264 \
#   -crf 23 \
#   -preset veryfast \
#   output1.mp4


# ffmpeg \
#   -i output1.mp4 \
#   -i sim.mp4 \
#   -filter_complex '[0:v]pad=iw*2:ih[int];[int][1:v]overlay=W/2:0[vid]' \
#   -map "[vid]" \
#   -c:v libx264 \
#   -crf 23 \
#   -preset veryfast \
#   real-sim+-sim.mp4

