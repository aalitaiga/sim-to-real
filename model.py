from collections import OrderedDict
import sys

from blocks.algorithms import GradientDescent, Adam, RMSProp
from blocks.bricks import Rectifier, Softmax, MLP
from blocks.bricks.cost import BinaryCrossEntropy, CategoricalCrossEntropy
from blocks.extensions import FinishAfter, Printing, ProgressBar
from blocks.extensions.stopping import FinishIfNoImprovementAfter
from blocks.extensions.monitoring import DataStreamMonitoring, TrainingDataMonitoring
from blocks.extensions.saveload import Checkpoint, Load
from blocks.graph import ComputationGraph
from blocks.initialization import IsotropicGaussian, Constant
from blocks.main_loop import MainLoop
from blocks.model import Model

import numpy as np
import theano
from theano import tensor as T

sys.setrecursionlimit(500000)

n_input = None
n_output = None
n_layer = 2
h = 128

x = T.matrix('features')
y = T.matrix('transition')

G = MLP(
    activations=[Rectifier(), Rectifier(),],
    dims=[n_input, h, h, n_output],
    weights_init=IsotropicGaussian(std=0.05, mean=0),
    biases_init=Constant(0.02)
)
D = MLP(
    activations=[Rectifier(), Rectifier(),],
    dims=[n_input, h, h, n_output],
    weights_init=IsotropicGaussian(std=0.05, mean=0),
    biases_init=Constant(0.02)
)
G.initialize()
D.initialize()

y_tilde = G.apply(x)

input_D_real = tensor.concatenate([x, y])
input_D_fake = tensor.concatenate([x, y_tilde])

pred_real = D.apply(input_D_real)
pred_fake = D.apply(input_D_fake)

discriminator_loss = (tensor.nnet.softplus(-data_preds) +
                      tensor.nnet.softplus(sample_preds)).mean()
generator_loss = (tensor.nnet.softplus(data_preds) +
                  tensor.nnet.softplus(-sample_preds)).mean()

gradients = OrderedDict()
gradients.update(
    zip(discriminator_parameters,
        theano.grad(discriminator_loss, discriminator_parameters)))
gradients.update(
    zip(generator_parameters,
        theano.grad(generator_loss, generator_parameters)))
step_rule = CompositeRule([Restrict(discriminator_step_rule,
                                    discriminator_parameters),
                           Restrict(generator_step_rule,
                                    generator_parameters)])
return GradientDescent(
    cost=generator_loss + discriminator_loss,
    gradients=gradients,
    parameters=discriminator_parameters + generator_parameters,
    step_rule=step_rule,
    on_unused_sources='ignore'
)

extensions = [
    FinishAfter(after_n_epochs=nb_epoch),
    FinishIfNoImprovementAfter(notification_name='test_pixelcnn_cost', epochs=patience),
    TrainingDataMonitoring(
        [algorithm.cost],
        prefix="train",
        after_epoch=True),
    DataStreamMonitoring(
        [algorithm.cost],
        test_stream,
        prefix="test"),
    Printing(),
    ProgressBar(),
    Checkpoint(path, every_n_epochs=save_every),
]

main_loop = MainLoop(
    algorithm=algorithm,
    data_stream=training_stream,
    model=model,
    extensions=extensions
)
main_loop.run()
