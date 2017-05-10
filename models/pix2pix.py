""" Generative model based on pix2pix https://arxiv.org/abs/1611.07004 """

from blocks.algorithms import Adam, RMSProp
from blocks.bricks import Identity, Logistic
from blocks.bricks.recurrent import RecurrentStack
from blocks.initialization import IsotropicGaussian, Constant

import numpy as np
import theano
from theano import tensor as T

from models.lstm import ConvLSTM, Linear2, Discriminator
from models.gan import RecurrentCGAN

n_lstm = 2
filter_size = (3,3)
num_channels = 3
num_filters = [64, 128, 256, 512, 512, 512]
discriminator_num_filters = [64, 128, 256, 512, 512, 512]
batch_size = 32
border_mode = 'half'
image_size = [(int(64 / 2**i),int(64 / 2**i)) for i in range(7)]
step = (2,2)
alpha = 10

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
    ## Decoder
    # feature map is (1, 1)
    ConvLSTM(
        filter_size, num_filters[-1], num_filters[5],
        image_size=image_size[-1], step=step, border_mode='half',
        batch_size=batch_size, convolution_type='deconv'
    ),
    # feature map is (2, 2)
    ConvLSTM(
        filter_size, num_filters[-2], num_filters[-1],
        image_size=image_size[-2], step=step, border_mode='half',
        batch_size=batch_size, convolution_type='deconv'
    ),
    # feature map is (4, 4)
    ConvLSTM(
        filter_size, num_filters[-3], num_filters[-2],
        image_size=image_size[-3], step=step, border_mode='half',
        batch_size=batch_size, convolution_type='deconv'
    ),
    # feature map is (8, 8)
    ConvLSTM(
        filter_size, num_filters[-4], num_filters[-3],
        image_size=image_size[-4], step=step, border_mode='half',
        batch_size=batch_size, convolution_type='deconv'
    ),
    # feature map is (16, 16)
    ConvLSTM(
        filter_size, num_filters[-5], num_filters[-4],
        image_size=image_size[-5], step=step, border_mode='half',
        batch_size=batch_size, convolution_type='deconv'
    ),
    # feature map is (32, 32)
    ConvLSTM(
        filter_size, num_filters[-6], num_filters[-5],
        image_size=image_size[-6], step=step, border_mode='half',
        batch_size=batch_size, convolution_type='deconv'
    ),
    # feature map is (64, 64)
    ConvLSTM(
        filter_size, num_channels, num_filters[-6],
        image_size=image_size[-7], step=(1,1), border_mode='half',
        batch_size=batch_size,
    ),
    ],
    fork_prototype=Identity(),
    weights_init=IsotropicGaussian(std=0.05, mean=0),
    biases_init=Constant(0.02),
    name='generator'
)

discriminator_rnn = RecurrentStack(
    [
    # input is (3, 64, 64)
    ConvLSTM(
        filter_size, discriminator_num_filters[0], num_channels,
        image_size=image_size[0], step=step, border_mode='half',
        batch_size=batch_size,
    ),
    # feature map is (32, 32)
    ConvLSTM(
        filter_size, discriminator_num_filters[1], discriminator_num_filters[0],
        image_size=image_size[1], step=step, border_mode='half',
        batch_size=batch_size,
    ),
    # feature map is (16, 16)
    ConvLSTM(
        filter_size, discriminator_num_filters[2], discriminator_num_filters[1],
        image_size=image_size[2], step=step, border_mode='half',
        batch_size=batch_size,
    ),
    # feature map is (8, 8)
    ConvLSTM(
        filter_size, discriminator_num_filters[3], discriminator_num_filters[2],
        image_size=image_size[3], step=step, border_mode='half',
        batch_size=batch_size,
    ),
    # feature map is (4, 4)
    ConvLSTM(
        filter_size, discriminator_num_filters[4], discriminator_num_filters[3],
        image_size=image_size[4], step=step, border_mode='half',
        batch_size=batch_size,
    ),
    # feature map is (2, 2)
    ConvLSTM(
        filter_size, discriminator_num_filters[5], discriminator_num_filters[4],
        image_size=image_size[5], step=step, border_mode='half',
        batch_size=batch_size,
    ),
    ],
    fork_prototype=Identity(),
)

linear = Linear2(
    input_dim=discriminator_num_filters[-1],
    output_dim=1, batch_size=batch_size,
    time=5
)

discriminator = Discriminator(
    discriminator_rnn,
    linear,
    weights_init=IsotropicGaussian(std=0.05, mean=0),
    biases_init=Constant(0.02),
)


if __name__ == '__main__':

    gan = RecurrentCGAN(generator, discriminator, alpha=alpha)
    gan.initialize()

    x = T.tensor5('target_sequence')
    y = T.tensor5('source_sequence')
    dis_loss, gen_loss = gan.losses(x, y)
    gen = gan.generator.apply(x)
    f = theano.function([x, y], [dis_loss, gen_loss])
    # dis_function = theano.function([x], discriminator.apply(x))

    a = np.random.rand(5, batch_size, 3, 64, 64).astype('float32')
    b = np.random.rand(5, batch_size, 3, 64, 64).astype('float32')
    # d = dis_function(a)
    cu = f(a, b)
    import ipdb; ipdb.set_trace()
