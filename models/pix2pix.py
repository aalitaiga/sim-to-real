""" Generative model based on pix2pix https://arxiv.org/abs/1611.07004 """

from blocks.algorithms import Adam, RMSProp
from blocks.bricks import Tanh, Identity, LeakyRectifier
from blocks.bricks.recurrent import RecurrentStack
from blocks.initialization import IsotropicGaussian, Constant

import numpy as np
import theano
from theano import tensor as T

from models.lstm import ConvLSTM, DEConvLSTM
from models.gan import GAN

n_lstm = 2
filter_size = (3,3)
num_channels = 3
num_filters = [64, 128, 256, 512, 512, 512]
batch_size = 32
border_mode = 'half'
image_size = [(int(64 / 2**i),int(64 / 2**i)) for i in range(7)]
step = (2,2)
alpha = 10

x = T.tensor5('features')

generator = RecurrentStack(
    [
    ## Encoder
    # input is (3, 64, 64)
    ConvLSTM(
        filter_size, num_filters[0], num_channels,
        image_size=image_size[0], step=step, border_mode='half',
        batch_size=batch_size
    ),
    # feature map is (32, 32)
    ConvLSTM(
        filter_size, num_filters[1], num_filters[0],
        image_size=image_size[1], step=step, border_mode='half',
        batch_size=batch_size
    ),
    # feature map is (16, 16)
    ConvLSTM(
        filter_size, num_filters[2], num_filters[1],
        image_size=image_size[2], step=step, border_mode='half',
        batch_size=batch_size
    ),
    # feature map is (8, 8)
    ConvLSTM(
        filter_size, num_filters[3], num_filters[2],
        image_size=image_size[3], step=step, border_mode='half',
        batch_size=batch_size
    ),
    # feature map is (4, 4)
    ConvLSTM(
        filter_size, num_filters[4], num_filters[3],
        image_size=image_size[4], step=step, border_mode='half',
        batch_size=batch_size
    ),
    # feature map is (2, 2)
    ConvLSTM(
        filter_size, num_filters[5], num_filters[4],
        image_size=image_size[5], step=step, border_mode='half',
        batch_size=batch_size
    ),
    # feature map is (1, 1)
    # ConvLSTM(
    #     filter_size, num_filters[4], num_filters[3],
    #     image_size=image_size[6], step=step, border_mode='half',
    #     batch_size=batch_size
    # )
    ## Decoder
    ## Encoder
    # feature map is (1, 1)
    DEConvLSTM(
        filter_size, num_filters[-1], num_filters[5],
        image_size=image_size[-1], step=step, border_mode='half',
        batch_size=batch_size
    ),
    # feature map is (2, 2)
    DEConvLSTM(
        filter_size, num_filters[-2], num_filters[-1],
        image_size=image_size[-2], step=step, border_mode='half',
        batch_size=batch_size
    ),
    # feature map is (4, 4)
    DEConvLSTM(
        filter_size, num_filters[-3], num_filters[-2],
        image_size=image_size[-3], step=step, border_mode='half',
        batch_size=batch_size
    ),
    # feature map is (8, 8)
    DEConvLSTM(
        filter_size, num_filters[-4], num_filters[-3],
        image_size=image_size[-4], step=step, border_mode='half',
        batch_size=batch_size
    ),
    # feature map is (16, 16)
    DEConvLSTM(
        filter_size, num_filters[-5], num_filters[-4],
        image_size=image_size[-5], step=step, border_mode='half',
        batch_size=batch_size
    ),
    # feature map is (32, 32)
    DEConvLSTM(
        filter_size, num_filters[-6], num_filters[-5],
        image_size=image_size[-6], step=step, border_mode='half',
        batch_size=batch_size
    ),
    # feature map is (64, 64)
    DEConvLSTM(
        filter_size, num_filters[-7], num_filters[-6],
        image_size=image_size[-7], step=step, border_mode='half',
        batch_size=batch_size
    )
    ],
    fork_prototype=Identity(),
    weights_init=IsotropicGaussian(std=0.05, mean=0),
    biases_init=Constant(0.02),
)

discriminator = ConvolutionalSequence([
    Convolutional((4,4), 64, name='conv_1'), LeakyRectifier(0.2),
    Convolutional((4,4), 128, name='conv_2'), SpatialBatchNormalization(), LeakyRectifier(0.2),
    Convolutional((4,4), 256, name='conv_3'), SpatialBatchNormalization(), LeakyRectifier(0.2),
    Convolutional((4,4), 512, name='conv_4'), SpatialBatchNormalization(), LeakyRectifier(0.2),
    Convolutional((4,4), 512, name='conv_5'), SpatialBatchNormalization(), LeakyRectifier(0.2),
])

gan = GAN(generator, discriminator, alpha=alpha)
