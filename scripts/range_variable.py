from fuel.streams import DataStream
from fuel.schemes import SequentialScheme
from fuel.datasets.hdf5 import H5PYDataset

import numpy as np

DATASET_PATH_REL = "/data/lisa/data/sim2real/"
DATASET_PATH = DATASET_PATH_REL + "mujoco_data_pusher3dof_big_backl.h5"
# DATASET_PATH = "/Tmp/alitaiga/mujoco_data_pusher3dof_big_backl.h5"
batch_size = 1
train_data = H5PYDataset(
    DATASET_PATH, which_sets=('train',), sources=('s_transition_obs','r_transition_obs', 'obs', 'actions')
)
stream_train = DataStream(train_data, iteration_scheme=SequentialScheme(train_data.num_examples, train_data.num_examples))
valid_data = H5PYDataset(
    DATASET_PATH, which_sets=('valid',), sources=('s_transition_obs','r_transition_obs', 'obs', 'actions')
)
stream_valid = DataStream(valid_data, iteration_scheme=SequentialScheme(train_data.num_examples, batch_size))

iterator = stream_train.get_epoch_iterator(as_dict=True)

data = next(iterator)
import ipdb; ipdb.set_trace()

## Max, min  mean std obs:
[2.56284189, 2.3500247, 2.39507723, 7.40329409, 9.20471668, 15.37792397]
[-0.12052701, -0.17479207, -0.73818409, -1.25026512, -3.95599413, -4.73272848]
[2.25090551, 1.94997263, 1.6495719, 0.43379614, 0.3314755, 0.43763939]
[0.5295766 ,  0.51998389,  0.57609886,  1.35480666, 1.40806067, 2.43865967]


# max_obs = np.zeros((6,))
# max_sim = np.zeros((6,))
# max_real = np.zeros((6,))
# max_act = np.zeros((3,))
# max_cor = np.zeros((6,))
# j = 0
#
# for i, data in enumerate(iterator):
#     # import ipdb; ipdb.set_trace()
#     max_obs = np.maximum(max_sim, data["obs"][0,:,:6].max(axis=0))
#     max_sim = np.maximum(max_sim, data["s_transition_obs"][0,:,:6].max(axis=0))
#     max_real = np.maximum(max_real, data["r_transition_obs"][0,:,:6].max(axis=0))
#     # max_cor = np.maximum(max_cor, (data["s_transition_obs"][0,:,:6] - data["r_transition_obs"][0,:,:6]).max(axis=0))
#     max_act = np.maximum(max_act, data["actions"][0,:,:].max(axis=0))


# print(max_obs)  # [2.56284189 2.3500247 2.39507723 7.40329409 9.20471668 15.37792397]
# print(max_sim)
# print(max_real)
# print(max_act)
# print(max_cor)  # [0.04541349, 0.15085796  0.1063659   1.4267087   8.24361801  6.95485973]

# mean_obs = np.zeros((6,))
# mean_sim = np.zeros((6,))
# mean_real = np.zeros((6,))
# mean_act = np.zeros((3,))
#
# for i, data in enumerate(iterator):
#     mean_obs += data["obs"][0,:,:6].sum(axis=0) / (100 * train_data.num_examples)
#     mean_real += data["r_transition_obs"][0,:,:6].sum(axis=0) / (100 * train_data.num_examples)
#     mean_sim += data["s_transition_obs"][0,:,:6].sum(axis=0) / (100 * train_data.num_examples)
#     mean_act += data["actions"][0,:,:].sum(axis=0) / (100 * train_data.num_examples)
#
# print(mean_obs)
# print(mean_sim)
# print(mean_real)
# print(mean_act)

min_obs = np.ones((6,))
min_sim = np.ones((6,))
min_real = np.ones((6,))
min_act = np.ones((3,))
min_cor = np.zeros((6,))

for i, data in enumerate(iterator):
    min_obs = np.minimum(min_sim, data["obs"][0,:,:6].min(axis=0))
    min_sim = np.minimum(min_sim, data["s_transition_obs"][0,:,:6].min(axis=0))
    min_real = np.minimum(min_real, data["r_transition_obs"][0,:,:6].min(axis=0))
    min_cor = np.minimum(min_cor, (data["s_transition_obs"][0,:,:6] - data["r_transition_obs"][0,:,:6]).min(axis=0))
    min_act = np.minimum(min_act, data["actions"][0,:,:].min(axis=0))

print(min_obs)
print(min_sim)
print(min_real)
print(min_act)
