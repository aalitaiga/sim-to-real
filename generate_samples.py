from blocks.extensions.saveload import Load
from fuel.streams import DataStream
from fuel.schemes import ShuffledScheme
from fuel.datasets.hdf5 import H5PYDataset
import numpy as np
import theano.tensor as T
import theano

main_loop = Load('Tmp/alitaiga/sim-to-real/pix2pix.tar')
import ipdb; ipdb.set_trace()
generator = main_loop.model.get_top_brick()[0].generator

img_source = (T.cast(T.tensor5('image_source')[:,:,:,64:192,64:192], 'float32') / 127.5) - 1
img_source = img_source.dimshuffle(1,0,2,3,4)

gen_func = theano.function(img_source, [generator.apply(img_source)])

robot_data = H5PYDataset(
    '/Tmp/alitaiga/sim-to-real/robot_data.h5',
    which_sets=('train',),
    sources=('image_source', 'image_target')
)
stream = DataStream(
    robot_data,
    iteration_scheme=ShuffledScheme(robot_data.num_examples, 1)
)

iterator = stream.get_epoch_iterator(as_dict=True)
sample = gen_func(next(iterator)['image_source'])

np.savez(
    '/u/alitaiga/repositories/samples',
    generated=sample
)
