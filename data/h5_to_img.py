import os

from fuel.streams import DataStream
from fuel.schemes import ShuffledScheme
from fuel.datasets.hdf5 import H5PYDataset
from scipy.misc import imsave

batch = 32
set_ = 'train'
# name = "/Tmp/alitaiga/sim-to-real/reacher_110k.h5"
# name = "/Tmp/alitaiga/sim-to-real/mujoco_data3.h5"
name = "/data/lisa/data/sim2real/mujoco_data3.h5"
out_dir_s = "/Tmp/alitaiga/sim-to-real/reacher_data/A/{}/".format(set_)
out_dir_r = "/Tmp/alitaiga/sim-to-real/reacher_data/B/{}/".format(set_)
reacher_data = H5PYDataset(name, which_sets=('train',),
                sources=('s_transition_img', 'r_transition_img'))

stream = DataStream(
    reacher_data,
    iteration_scheme=ShuffledScheme(reacher_data.num_examples, batch)
)

for out_dir in [out_dir_s, out_dir_r]:
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

iterator = stream.get_epoch_iterator(as_dict=True)

i = 0

for dic in iterator:
    # import ipdb; ipdb.set_trace()
    s_trans_img = dic['s_transition_img']
    r_trans_img = dic['r_transition_img']
    seq_len = s_trans_img.shape[1]

    for j in range(batch):
        for k in range(seq_len):
            try:
                imsave(out_dir_s + "{}.png".format(str(i)), s_trans_img[j, k, :, :, :])
                imsave(out_dir_r + "{}.png".format(str(i)), r_trans_img[j, k, :, :, :])
                i += 1
            except IndexError:
                continue

            if i % 5000 == 0:
                print(i)
