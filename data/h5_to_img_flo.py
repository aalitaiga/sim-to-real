import os
from fuel.streams import DataStream
from fuel.schemes import ShuffledScheme
from fuel.datasets.hdf5 import H5PYDataset
from scipy.misc import imsave
from tqdm import tqdm

batch = 1
set_ = 'train'
# name = "/Tmp/alitaiga/sim-to-real/reacher_110k.h5"
# name = "/Tmp/alitaiga/sim-to-real/gen_data.h5"
# out_dir_s = "/Tmp/alitaiga/sim-to-real/reacher_data/A/{}/".format(set_)
# out_dir_r = "/Tmp/alitaiga/sim-to-real/reacher_data/B/{}/".format(set_)
name = "./mujoco_data.h5"
out_dir_s = "/tmp/sim-to-real/reacher_data/A/{}/".format(set_)
out_dir_r = "/tmp/sim-to-real/reacher_data/B/{}/".format(set_)

for out_dir in [out_dir_r, out_dir_s]:
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

reacher_data = H5PYDataset(name, which_sets=(set_,),
                sources=('s_transition_img', 'r_transition_img'))

stream = DataStream(
    reacher_data,
    iteration_scheme=ShuffledScheme(reacher_data.num_examples, batch)
)

iterator = stream.get_epoch_iterator(as_dict=True)

episode_idx = 0
for episode in tqdm(iterator):
    # import ipdb; ipdb.set_trace()
    s_trans_img = episode['s_transition_img'][0]
    r_trans_img = episode['r_transition_img'][0]

    out_dir_s_episode = os.path.join(out_dir_s, "run-{}".format(episode_idx))
    out_dir_r_episode = os.path.join(out_dir_r, "run-{}".format(episode_idx))

    for out_dir in [out_dir_s_episode, out_dir_r_episode]:
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

    for frame_idx in range(len(s_trans_img)):
        imsave(os.path.join(out_dir_s_episode, "{}.png".format(frame_idx)), s_trans_img[frame_idx, :, :, :])
        imsave(os.path.join(out_dir_r_episode, "{}.png".format(frame_idx)), r_trans_img[frame_idx, :, :, :])

    episode_idx += 1
