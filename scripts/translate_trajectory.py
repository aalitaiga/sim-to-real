from fuel.streams import DataStream
from fuel.schemes import ShuffledScheme
from fuel.datasets.hdf5 import H5PYDataset
import imageio
import gym
import gym_reacher2
from scipy.misc import imresize, imsave
from torch.autograd import Variable
from torch import torch
import torchvision.transforms as transforms
import numpy as np

from cyclegan_pix2pix.options.test_options import TestOptions
from models.models import create_model

# You need to run the data on your local machine to be able
# to use Mujoco data
# If you can't have access to a gpu get the data first on your machine
# Then convert it on other machine with a gpu

opt = TestOptions().parse()
# Cycle gan parameters
opt.model = 'cycle_gan'
opt.name = 'reacher_cyclegan_ball50_resnet_6blocks'
opt.checkpoints_dir = '/u/alitaiga/repositories/cyclegan_pix2pix/checkpoints/'
opt.dataset_mode = 'unaligned'
opt.which_epoch = 'latest'
opt.dataroot = '/data/lisa/data/sim2real/mujoco_data4.h5'
# opt.no_dropout = False
# opt.identity = 0.1
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip
opt.add_ball = True
opt.which_model_netG = 'resnet_6blocks'
opt.resize_or_crop = False
# opt.gpu_ids = [0]
# # run on cpu
# opt.gpu_ids = -1
args = vars(opt)
print('------------ Options -------------')
for k, v in sorted(args.items()):
    print('%s: %s' % (str(k), str(v)))
print('-------------- End ----------------')
# import ipdb; ipdb.set_trace()
model = create_model(opt)

# # Create the two envs:
# env = gym.make('Reacher2Pixel-v1')
# env2 = gym.make('Reacher2Pixel-v1')
#
# env.env.env._init(
#     arm0=.1,    # length of limb 1
#     arm1=.1,     # length of limb 2
#     torque0=200, # torque of joint 1
#     torque1=200  # torque of joint 2
# )
# env2.env.env._init(
#     arm0=.12,    # length of limb 1
#     arm1=.08,     # length of limb 2
#     torque0=200, # torque of joint 1
#     torque1=200,  # torque of joint 2
#     fov=50,
#     colors={
#         "arenaBackground": ".27 .27 .81",
#         "arenaBorders": "1.0 0.8 0.4",
#         "arm0": "0.2 0.6 0.2",
#         "arm1": "0.2 0.6 0.2"
#     }
# )
#
# def match_env(ev1, ev2):
#     # make env1 state match env2 state (simulator matches real world)
#     ev1.env.env.set_state(
#         ev2.env.env.model.data.qpos.ravel(),
#         ev2.env.env.model.data.qvel.ravel()
#     )

images_env1 = []
images_env2 = []
images_translated = []
n_trajectory = 1
transform = transforms.Normalize((0.5, 0.5, 0.5),
                     (0.5, 0.5, 0.5))

batch = 1
name = "/data/lisa/data/sim2real/mujoco_data4.h5"
reacher_data = H5PYDataset(name, which_sets=('valid',),
                sources=('s_transition_img', 'r_transition_img'))

stream = DataStream(
    reacher_data,
    iteration_scheme=ShuffledScheme(reacher_data.num_examples, batch)
)

i = 0
for data in stream.get_epoch_iterator(as_dict=True):
    s_trans_img = data['s_transition_img'][0]
    r_trans_img = data['r_transition_img'][0]
    in_ = torch.from_numpy(s_trans_img).float().cuda()
    in_ = in_.permute(0,3,1,2)
    in_ = transform(in_)
    out = model.netG_A.forward(Variable(in_, volatile=True))
    out = out.data.permute(0,2,3,1).cpu() * 128 + 128
    out = out.byte().numpy()
    images_translated = [out[j, :, :, :] for j in range(150)]

    imageio.mimsave('env1_{}.gif'.format(i), [s_trans_img[j,:,:,:] for j in range(150)])
    imageio.mimsave('env2_{}.gif'.format(i), [r_trans_img[j,:,:,:] for j in range(150)])
    imageio.mimsave('translated_{}.gif'.format(i), images_translated)
    i = i + 1

    if i == 15:
        break

# imageio.mimsave('env1.gif', [s_trans_img[i,:,:,:] for i in range(150)])
# imageio.mimsave('env2.gif', [r_trans_img[i,:,:,:] for i in range(150)])
# imageio.mimsave('translated.gif', images_translated)
