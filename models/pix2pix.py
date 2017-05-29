""" Generative model based on pix2pix https://arxiv.org/abs/1611.07004 """
from datetime import datetime
import logging
import math
import os
import sys

from blocks.algorithms import Adam
from blocks.bricks import Identity
from blocks.bricks.recurrent import RecurrentStack
from blocks.initialization import IsotropicGaussian, Constant
from fuel.streams import DataStream
from fuel.schemes import ShuffledScheme
from fuel.datasets.hdf5 import H5PYDataset

import theano
from theano import tensor as T

from utils.extensions import VisdomExt, GenerateSamples
# from utils.original_main_loop import MainLoop
from blocks.main_loop import MainLoop
from models.lstm import ConvLSTM, Linear2, Discriminator
from models.gan import RecurrentCGAN

logging.basicConfig()

sys.setrecursionlimit(500000)

filter_size = (3,3)
num_channels = 3
num_filters = [3, 8, 16, 32, 32, 64, 64, 64] #[3, 64, 128, 256, 512, 512, 512, 512]
discriminator_num_filters = [3, 8, 16, 32, 32, 64, 64, 64]  #[3, 64, 128, 256, 512, 512, 512, 512]
batch_size = 1
border_mode = 'half'
img_dim = 128
image_size = [(int(img_dim / 2**i),int(img_dim / 2**i)) for i in range(int(math.log(img_dim, 2))+1)]
step = (2,2)
alpha = 10
assert len(num_filters) == len(image_size) == len(discriminator_num_filters)

encoder_archi = [ConvLSTM(
    filter_size, num_filters[i+1], num_filters[i],
    image_size=image_size[i], step=step, border_mode='half',
    batch_size=batch_size, name='encoder_lstm', weightnorm=True
) for i in range(len(image_size)-1)]
decoder_archi = [ConvLSTM(
    filter_size, num_filters[-i-1], num_filters[-i],
    image_size=image_size[-i], step=step, border_mode='half',
    batch_size=batch_size, convolution_type='deconv', weightnorm=True,
    name='decoder_lstm'
) for i in range(1,len(image_size))]

generator = RecurrentStack(
    encoder_archi + decoder_archi,
    fork_prototype=Identity(name='identity'),
    weights_init=IsotropicGaussian(std=0.05, mean=0),
    biases_init=Constant(0.02),
    name='generator'
)

discriminator_rnn = RecurrentStack(
    [
        ConvLSTM(
            filter_size, discriminator_num_filters[i+1], discriminator_num_filters[i],
            image_size=image_size[i], step=step, border_mode='half',
            batch_size=batch_size, weightnorm=True
        ) for i in range(len(image_size)-1)
    ],
    fork_prototype=Identity(),
)

linear = Linear2(
    input_dim=discriminator_num_filters[-1],
    output_dim=1, batch_size=batch_size,
    time=60
)

discriminator = Discriminator(
    discriminator_rnn,
    linear,
    weights_init=IsotropicGaussian(std=0.05, mean=0),
    biases_init=Constant(0.02),
)


if __name__ == '__main__':

    from blocks.extensions import FinishAfter, Printing, ProgressBar
    from blocks.extensions.monitoring import TrainingDataMonitoring
    from blocks.extensions.saveload import Checkpoint, Load
    from blocks.graph import apply_noise, apply_dropout, ComputationGraph
    from blocks.model import Model
    # from blocks.main_loop import MainLoop
    from blocks.filter import VariableFilter
    from blocks.roles import INPUT

    path = '/Tmp/alitaiga/sim-to-real/{}'.format(datetime.now())
    if not os.path.exists(path):
        os.makedirs(path)

    lr = 1e-4
    beta1 = 0.5
    dropout = 0.5
    load = False
    noise_std = 0.2

    gan = RecurrentCGAN(generator, discriminator, alpha=alpha)
    gan.initialize()

    robot_data = H5PYDataset(
        '/Tmp/alitaiga/sim-to-real/robot_data.h5',
        which_sets=('train',),
        sources=('image_source', 'image_target')
    )
    stream = DataStream(
        robot_data,
        iteration_scheme=ShuffledScheme(robot_data.num_examples, batch_size)
    )

    # Crop the image from 256x256 to 128, 128
    input_ = T.tensor5('image_source')
    img_source = (T.cast(input_[:,:,:,64:192,64:192], 'float32') / 127.5) - 1
    img_target = (T.cast(T.tensor5('image_target')[:,:,:,64:192,64:192], 'float32') / 127.5) - 1
    # acts = T.tensor3('actions')
    # states = T.tensor3('states')
    #
    # full_states = T.concatenate([states, acts], axis=2)

    # Change from (batch, time, bla, bla) to (time, batch, bla, bla)
    img_source = img_source.dimshuffle(1,0,2,3,4)
    img_target = img_target.dimshuffle(1,0,2,3,4)
    # full_states = full_states.dimshuffle(1,0,2)

    discriminator_loss, generator_loss = gan.losses(img_source, img_target)
    discriminator_loss.name = 'discriminator_loss'
    generator_loss.name = 'generator_loss'
    cg = ComputationGraph([discriminator_loss, generator_loss])

    if dropout > 0:
        # Be careful when skip connections will be added
        inputs = VariableFilter(roles=[INPUT])(cg.variables)
        to_keep = ['decoder_lstm#{}_apply_inputs'.format(i) for i in range(len(image_size)-1, len(image_size)+2)]
        dropout_inputs = [inp for inp in inputs if inp.name in to_keep]
        cg = apply_dropout(cg, dropout_inputs, dropout)

    if noise_std > 0:
        noisy_inputs = []
        cg = apply_noise(cg, noisy_inputs, noise_std)

    model = Model(cg.outputs)
    auxiliary_variables = [u for u in model.auxiliary_variables if 'norm' not in u.name]
    # auxiliary_variables.extend(VariableFilter(theano_name='gradient_penalty')(model.variables))
    auxiliary_variables.extend(VariableFilter(theano_name='abs_error')(model.variables))
    step_rule = Adam(learning_rate=lr, beta1=beta1)
    # discriminator_algo, generator_algo = gan.algorithm(discriminator_loss, generator_loss, step_rule, step_rule)
    algorithm = gan.algorithm(discriminator_loss, generator_loss, step_rule, step_rule)

    extensions = [
        FinishAfter(after_n_epochs=0),
        TrainingDataMonitoring(
            model.outputs+auxiliary_variables,
            # prefix="train",
            after_batch=True),
        ProgressBar(),
        VisdomExt([[
            'discriminator_loss',
            'generator_loss'
        ], [
            # 'gradient_penalty',
            'abs_error'
        ], [
            'data_accuracy', 'sample_accuracy'
        ]], [dict(title='WGAN losses', xlabel='iterations', ylabel='value`'),
            dict(title='Other costs', xlabel='iterations', ylabel='value'),
            dict(title='Discriminator accuracies', xlabel='iterations', ylabel='probability')],
            {'env': 'WGAN_{}'.format(datetime.now())}, after_batch=True),
        # Load('Tmp/alitaiga/sim-to-real/pix2pix.tar'),
        Checkpoint(path+'/pix2pix.tar', every_n_epochs=2, after_training=True,
                   use_cpickle=True),
        GenerateSamples(
            theano.function([input_], gan.generator.apply(img_source)),
            path, every_n_epochs=3),
        Printing()
    ]

    main_loop = MainLoop(
        algorithm,
        data_stream=stream,
        model=model,
        extensions=extensions,
        # generator_algorithm=generator_algo,
        # discriminator_iters=5
    )

    main_loop.run()
    import ipdb; ipdb.set_trace()
