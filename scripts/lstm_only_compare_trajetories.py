import gym
import gym_throwandpush
from gym.monitoring import VideoRecorder
import numpy as np
from torch import from_numpy, load
from torch.autograd import Variable
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pylab import rcParams

from simple_joints_lstm.pusher_lstm import LstmSimpleNet2Pusher

rcParams['axes.facecolor'] = 'white'
rcParams['figure.figsize'] = 15, 9
plt.style.use('ggplot')
np.random.seed(0)

means = {
    'o': np.array([-0.4417094, 1.50765455, -0.02639891, -0.05560728, 0.39159551, 0.03819341, 0.76052153, 0.23057458, 0.63315856, -0.6400153, 1.01691067, -1.02684915], dtype='float32'),
    's': np.array([-0.44221497, 1.52240622, -0.02244471, 0.01573334, 0.23615479, 0.10089023, 0.7594685, 0.23817146, 0.63317519, -0.64011943, 1.01691067, -1.02684915], dtype='float32'),
    # 'c': np.array([-2.23746197e-03, 4.93022148e-03, -2.03814497e-03, -6.97841570e-02, 1.53955221e-01, -6.21460043e-02], dtype='float32')
    'c': np.array([-2.73653027e-03, 1.95451882e-02, 1.91621704e-03, 1.56128232e-03, -1.51736499e-03, 5.49889286e-04, -1.03732746e-03, 7.78057473e-03, 1.73114568e-05, -9.45877109e-05], dtype='float32')
    # 'r': np.array([2.25277853,  1.95338345, 1.64534044, 0.48487723, 0.45031613, 0.30320421], dtype='float32')
}

std = {
    'o': np.array([0.38327965, 0.78956741, 0.48310387, 0.33454728, 0.53120506, 0.51319438, 0.20692779, 0.36664706, 0.25205335, 0.15865214, 0.11554158, 0.1132608], dtype='float32'),
    's': np.array([0.38500383, 0.78036022, 0.48781601, 0.35502997, 0.60374367, 0.56180185, 0.21046612, 0.36828887, 0.25209084, 0.15857539, 0.11554158, 0.1132608], dtype='float32'),
    # 'c': np.array([7.19802594e-03, 1.59114692e-02, 7.24539673e-03, 2.23035514e-01, 4.93483037e-01, 2.18238667e-01,], dtype='float32'),
    'c': np.array([0.01655727, 0.02646242, 0.02456561, 0.11201099, 0.1206677, 0.31924954, 0.00993428, 0.00796531, 0.00071493, 0.00133473], dtype='float32'),
    'a': np.array([0.57690412, 0.57732242, 0.57705152], dtype='float32')
    # 'r':  np.array([0.52004296, 0.51547343, 0.57784373, 1.30222356, 1.36113203, 2.38046765], dtype='float32')
}

env = gym.make("Pusher3Dof2-v0")
env.env._init(
    torques=[1, 1, 1],
    xml='3link_gripper_push_2d_backlash',
    colored=False
)
env.reset()

# env2 = gym.make("Pusher3Dof2-v0")
# env2.env.env._init(
#     torques=[1, 1, 1],
#     colored=True,
# )
# env2.reset()
net = LstmSimpleNet2Pusher(15, 10, use_cuda=False)
model = "/u/alitaiga/repositories/sim-to-real/trained_models/real_only/lstm_pusher_3l_128_best.pt"
checkpoint = load(model, map_location=lambda storage, loc: storage)
net.load_state_dict(checkpoint['state_dict'])


def match_env(env_1, env_2):
    # set env2 (simulator) to that of env1 (real robot)
    env_2.unwrapped.set_state(
        env_1.unwrapped.model.data.qpos.ravel(),
        env_1.unwrapped.model.data.qvel.ravel()
    )

def make_variables(obs, action):
    input_ = np.concatenate([
        (obs[None, None, :] - means['o']) / std['o'],
        (action[None, None, :] / std['a']),
    ], axis=2)
    return Variable(from_numpy(input_).float(), volatile=True)


# match_env(env, env2)
# video_recorder = VideoRecorder(
#     env, 'sim+backlash.mp4', enabled=True)
# video_recorder2 = VideoRecorder(
#     env2, 'sim+.mp4', enabled=True)

num_obs = 10
array = np.zeros((2, 100, num_obs))

for i, data in enumerate(range(1)):
    new_obs = env.reset()
    # env2.reset()
    # match_env(env, env2)
    obs2 = new_obs

    array[0, 0, :] = new_obs[:num_obs]
    array[1, 0, :] = obs2[:num_obs]

    for j in range(1, 100):
        # env.render()
        # env2.render()

        action = env.action_space.sample()
        new_obs, reward, done, info = env.step(action.copy())
        new_obs2_scaled = net.forward(make_variables(obs2.copy(), action.copy())).data.cpu().numpy()[0,0,:]
        new_obs2 = (new_obs2_scaled * std['c']) + means['c']
        new_obs2 = np.append(new_obs2, [0,0])

        # new_obs2, reward2, done2, info2 = env2.step(action.copy())
        array[0, j, :] = new_obs[:num_obs]
        array[1, j, :] = obs2[:num_obs] + new_obs2[:num_obs]

        obs2 = obs2 + new_obs2
        if done:
            break
    if i == 1:
        break
print(j)
# video_recorder.close()
# video_recorder.enabled = False
# video_recorder2.close()
# video_recorder2.enabled = False
cm_name ='tab10'
color_map = cm.get_cmap(cm_name)
colors = color_map([x/(float(num_obs)+2) for x in range(num_obs)])

x = np.arange(j+1)
for y in range(num_obs):
    plt.plot(x, array[0,:,y], color=colors[y])
    plt.plot(x, array[1,:,y], linestyle='--', color=colors[y])
plt.title("Forward simulation on Pusher3Dof")
plt.savefig('forward_lstm.jpg', bbox_inches='tight')
plt.show()
