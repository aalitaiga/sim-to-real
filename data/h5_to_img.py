from fuel.streams import DataStream
from fuel.schemes import ShuffledScheme
from fuel.datasets.hdf5 import H5PYDataset
from scipy.misc import imsave

batch = 32
set_ = 'val'
# name = "/Tmp/alitaiga/sim-to-real/reacher_110k.h5"
name = "/Tmp/alitaiga/sim-to-real/gen_data.h5"
out_dir_s = "/Tmp/alitaiga/sim-to-real/reacher_data/A/{}/".format(set_)
out_dir_r = "/Tmp/alitaiga/sim-to-real/reacher_data/B/{}/".format(set_)
reacher_data = H5PYDataset(name, which_sets=('valid',),
                sources=('s_transition_img', 'r_transition_img'))

stream = DataStream(
    reacher_data,
    iteration_scheme=ShuffledScheme(reacher_data.num_examples, batch)
)

iterator = stream.get_epoch_iterator(as_dict=True)

i = 0

for dic in iterator:
    # import ipdb; ipdb.set_trace()
    s_trans_img = dic['s_transition_img']
    r_trans_img = dic['r_transition_img']

    for j in range(batch):
        try:
            imsave(out_dir_s + "{}.png".format(str(i)), s_trans_img[j, :, :, :])
            imsave(out_dir_r + "{}.png".format(str(i)), r_trans_img[j, :, :, :])
            i += 1
        except IndexError:
            continue
