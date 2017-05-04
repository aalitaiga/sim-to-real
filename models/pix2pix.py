""" Generative model based on pix2pix https://arxiv.org/abs/1611.07004"""

from blocks.algorithms import Adam, RMSProp
from blocks.bricks import Tanh, Identity, LeakyRectifier
from blocks.bricks.bn import SpatialBatchNormalization
from blocks.bricks.conv import Convolutional
from blocks.bricks.sequences import Sequence
from blocks.initialization import IsotropicGaussian, Constant

from models.lstm import ConvLSTM
from model.gan import GAN


n_lstm = 3
dim = 4
filter_size = 3
num_channels = 3
num_filters = 16
batch_size = 32
border_mode = 'half'
image_size = (64,64)
step = (2,2)
alpha = 10

first_lstm = ConvLSTM(dim, n_lstm, filter_size, num_filters, num_channels,
batch_size=batch_size, image_size=image_size, step=step, border_mode=border_mode)
second_lstm = ConvLSTM(dim, n_lstm, filter_size, num_filters, num_channels,
batch_size=batch_size, image_size=image_size, step=step, border_mode=border_mode)

generator = Sequence(
    [first_lstm, second_lstm],
    weights_init=IsotropicGaussian(std=0.05, mean=0),
    biases_init=Constant(0.02)
)

discriminator = Sequence([
    Convolutional((4,4), 64, name='conv_1'), LeakyRectifier(0.2),
    Convolutional((4,4), 128, name='conv_2'), SpatialBatchNormalization(), LeakyRectifier(0.2),
    Convolutional((4,4), 256, name='conv_3'), SpatialBatchNormalization(), LeakyRectifier(0.2),
    Convolutional((4,4), 512, name='conv_4'), SpatialBatchNormalization(), LeakyRectifier(0.2),
    Convolutional((4,4), 512, name='conv_5'), SpatialBatchNormalization(), LeakyRectifier(0.2),
])

gan = GAN(generator, discriminator, alpha=alpha)
