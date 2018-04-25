import gym
import gym_throwandpush
from gym.monitoring import VideoRecorder
import numpy as np

from simple_joints_lstm.striker_lstm import LstmStriker

model = "/u/alitaiga/repositories/sim-to-real/trained_models/lstm_striker_3l_128_best_059.pt"

env_sim = gym.make('Striker-v1')  # sim
env_simplus = gym.make('StrikerPlus-v0')  # sim
env_real = gym.make('Striker-v2')  # real

env_sim.reset()

env_simplus.load_model(
    LstmStriker(53,14, use_cuda=False, hidden_nodes=128, lstm_layers=3, wdrop=0),
    model
)
env_simplus.reset()

env_real.reset()


def match_env(real, sim):
    # set env1 (simulator) to that of env2 (real robot)
    # simulator = sim.env
    # if hasattr(simulator, "env"):
    #     simulator = simulator.env

    simulator.unwrapped.set_state(
        real.unwrapped.model.data.qpos.ravel(),
        real.unwrapped.model.data.qvel.ravel()
    )


# dataset_train = MujocoTraintestPusherSimpleDataset(DATASET_PATH, for_training=True)
# dataloader_train = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True)

match_env(env_real, env_simplus)
match_env(env_real, env_sim)
video_recorder_real = VideoRecorder(
    env_real, 'real.mp4', enabled=True)
video_recorder_simplus = VideoRecorder(
    env_simplus, 'sim+.mp4', enabled=True)
video_recorder_sim = VideoRecorder(
    env_sim, 'sim.mp4', enabled=True)

video_recorders = [video_recorder_real, video_recorder_simplus, video_recorder_sim]

# for i, data in enumerate(dataloader_train):
for i in range(30):
    for j in range(100):
        env_real.render()
        env_simplus.render()
        env_sim.render()

        # action = data["actions"][0, j].numpy()
        action = env_sim.action_space.sample()

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
    if i == 17:
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


ffmpeg \
  -i output1.mp4 \
  -i sim.mp4 \
  -filter_complex '[0:v]pad=iw*2:ih[int];[int][1:v]overlay=W/2:0[vid]' \
  -map "[vid]" \
  -c:v libx264 \
  -crf 23 \
  -preset veryfast \
  real-sim+-sim.mp4
