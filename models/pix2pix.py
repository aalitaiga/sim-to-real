""" Generative model based on pix2pix https://arxiv.org/abs/1611.07004"""
from blocks.algorithms import Adam, RMSProp
from blocks.bricks import Tanh, Identity, LeakyRectifier
from blocks.bricks.bn import SpatialBatchNormalization
from blocks.bricks.conv import Convolutional, ConvolutionalSequence
from blocks.bricks.sequences import Sequence
from blocks.initialization import IsotropicGaussian, Constant

from fuel.datasets import CIFAR10
from fuel.streams import DataStream
from fuel.schemes import SequentialScheme
from fuel.transformers import Flatten

import numpy as np
import theano
from theano import tensor as T

from models.lstm import ConvLSTM
from models.gan import GAN


n_lstm = 3
filter_size = (3,3)
num_channels = 3
num_filters = 16
batch_size = 32
border_mode = 'half'
image_size = [(int(64 / 2**i),int(64 / 2**i)) for i in range(7)]
step = (2,2)
alpha = 10

x = T.tensor5('features')
first_lstm = ConvLSTM(filter_size, num_filters, num_channels,
    image_size=(64,64), step=step, border_mode='half',
    weights_init=IsotropicGaussian(std=0.05, mean=0),
    biases_init=Constant(0.02), batch_size=batch_size
)
first_lstm.initialize()
h = first_lstm.apply(x)
f = theano.function([x], h)

array = np.ones((1, batch_size, 3, 64, 64))
array = np.concatenate([array, 2*array, 3*array], axis=0).astype('float32')
assert array.shape[0] == 3
ans = f(array)

# x_ = T.tensor4('features')
# conv = Convolutional(
#     filter_size, num_filters, num_channels,
#     batch_size=batch_size, image_size=(64,64),
#     step=step, border_mode=border_mode,
#     weights_init=IsotropicGaussian(std=0.05, mean=0),
#     biases_init=Constant(0.02),
#     name='convolution_input'
# )
# conv.initialize()
# h_ = conv.apply(x_)
# f = theano.function([x_], h_)
#
# array = np.ones((batch_size, 3, 64, 64)).astype('float32')
# ans =  f(array)
# import ipdb; ipdb.set_trace()
# training_stream = DataStream(
#     data,
#     iteration_scheme=ShuffledScheme(data.num_examples, batch_size)
# )



# cifar = CIFAR10(("train",))



import ipdb; ipdb.set_trace()
# second_lstm = ConvLSTM(dim, n_lstm, filter_size, num_filters,
# image_size=image_size, step=step)
#
# generator = ConvolutionalSequence(
#     [
#         ConvLSTM(dim, n_lstm, filter_size, num_filters,
#         image_size=image_size[0], step=step),
#         ConvLSTM(dim, n_lstm, filter_size, num_filters,
#         image_size=image_size[1], step=step)
#         ConvLSTM(dim, n_lstm, filter_size, num_filters,
#         image_size=image_size[2], step=step)
#         ConvLSTM(dim, n_lstm, filter_size, num_filters,
#         image_size=image_size[3], step=step)
#         ConvLSTM(dim, n_lstm, filter_size, num_filters,
#         image_size=image_size[4], step=step)
#         ConvLSTM(dim, n_lstm, filter_size, num_filters,
#         image_size=image_size[5], step=step)
#         ConvLSTM(dim, n_lstm, filter_size, num_filters,
#         image_size=image_size[6], step=step)
#         ConvLSTM(dim, n_lstm, filter_size, num_filters,
#         image_size=image_size, step=step)
#         ConvLSTM(dim, n_lstm, filter_size, num_filters,
#         image_size=image_size, step=step)
#     ],
#     num_channels=num_channels,
#     weights_init=IsotropicGaussian(std=0.05, mean=0),
#     biases_init=Constant(0.02),
#     batch_size=batch_size,
#     border_mode=border_mode,
# )
#
# discriminator = ConvolutionalSequence([
#     Convolutional((4,4), 64, name='conv_1'), LeakyRectifier(0.2),
#     Convolutional((4,4), 128, name='conv_2'), SpatialBatchNormalization(), LeakyRectifier(0.2),
#     Convolutional((4,4), 256, name='conv_3'), SpatialBatchNormalization(), LeakyRectifier(0.2),
#     Convolutional((4,4), 512, name='conv_4'), SpatialBatchNormalization(), LeakyRectifier(0.2),
#     Convolutional((4,4), 512, name='conv_5'), SpatialBatchNormalization(), LeakyRectifier(0.2),
# ])
#
# gan = GAN(generator, discriminator, alpha=alpha)
